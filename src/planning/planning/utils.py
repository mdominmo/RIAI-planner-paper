import copy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import math
import transforms3d.quaternions as tq
from geographiclib.geodesic import Geodesic
from transforms3d.euler import euler2quat
import csv
from ament_index_python.packages import get_package_share_directory
import os
from scipy import interpolate


def model_static_obstacles(
        obstacles_poses,
        t_final,
        cylinder_height,
        obstacle_radius
):
    obstacles = []
    for obstacle in obstacles_poses:
        delta_t = .5
        t_obs = np.arange(0, t_final + delta_t, delta_t)
        n_points = len(t_obs)
        x_obs = np.full(n_points, obstacle.position.x)
        y_obs = np.full(n_points, obstacle.position.y)
        z_obs = np.full(n_points, cylinder_height)
        r_obs = np.full(n_points, obstacle_radius)

        obstacle = np.column_stack([x_obs, y_obs, z_obs, t_obs, r_obs])
        obstacles.append(obstacle)
    
    return obstacles


def densify_waypoints(poses, points_per_segment=5):
    """
    Genera puntos intermedios lineales entre waypoints para suavizar el spline.
    
    Args:
        poses (list[Pose]): waypoints originales.
        points_per_segment (int): número de puntos extra por segmento.
    
    Returns:
        list[Pose]: lista de poses con puntos extra.
    """
    new_poses = []

    for i in range(len(poses)-1):
        p0 = poses[i]
        p1 = poses[i+1]

        for j in range(points_per_segment):
            t = j / points_per_segment
            p = Pose()
            p.position.x = (1-t)*p0.position.x + t*p1.position.x
            p.position.y = (1-t)*p0.position.y + t*p1.position.y
            p.position.z = (1-t)*p0.position.z + t*p1.position.z
            new_poses.append(p)
    
    # Añadir el último waypoint original
    new_poses.append(poses[-1])

    return new_poses


def bspline_trajectory(poses, n_points=200, smoothing=0.0, degree=3, periodic=False, speed=1.0):
    """
    Genera una trayectoria suavizada mediante B-spline.
    Devuelve una lista de diccionarios con Pose, yaw y vector de velocidad (vx,vy,vz),
    y el tiempo medio entre puntos de la trayectoria.
    """

    result = []

    if len(poses) == 1:
        result.append({
            "pose": poses[0],
            "yaw": float('nan'),
            "vel": float('nan'),
            "dt": float('nan')
        })
        return result
    else: 
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in densify_waypoints(poses, 5)]).T
        
        tck, u = interpolate.splprep(pts, s=smoothing, k=degree, per=periodic)
        u_new = np.linspace(0, 1, n_points)
        out = interpolate.splev(u_new, tck)         
        dout = interpolate.splev(u_new, tck, der=1) 

        total_length = 0.0

        for i in range(n_points):
            x, y, z = out[0][i], out[1][i], out[2][i]
            dx, dy, dz = dout[0][i], dout[1][i], dout[2][i]

            yaw = math.atan2(dy, dx)
            norm = math.sqrt(dx**2 + dy**2 + dz**2)
            if norm > 1e-6:
                vx, vy, vz = (dx / norm * speed,
                            dy / norm * speed,
                            dz / norm * speed)
            else:
                vx, vy, vz = 0.0, 0.0, 0.0

            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = z

            vel = Twist()   
            vel.linear.x = vx
            vel.linear.y = vy 
            vel.linear.z = vz
            
            if i > 0:
                dx_seg = out[0][i] - out[0][i-1]
                dy_seg = out[1][i] - out[1][i-1]
                dz_seg = out[2][i] - out[2][i-1]
                seg_length = math.sqrt(dx_seg**2 + dy_seg**2 + dz_seg**2)
                dt = seg_length / speed

                result.append({
                    "pose": p,
                    "yaw": yaw,
                    "vel": vel,
                    "dt": dt
                })

        return result


def enu_ned_trajectories(trajectories):
    for traj in trajectories:
        ned_pos = []
        for p in traj[0]:
            ned_pos.append(enu_ned(p))
        traj[0] = ned_pos
    return trajectories


def gps_offset_in_enu(lat_ref, lon_ref, alt_ref, lat_origin, lon_origin, alt_origin):
        # Calcula distancia y azimut entre ref y origin usando geodesia
        geod = Geodesic.WGS84
        g = geod.Inverse(lat_origin, lon_origin, lat_ref, lon_ref)
        distance = g['s12']  # distancia en metros entre los dos puntos

        # Azimut (desde origen hacia ref)
        azimuth = g['azi1']

        # Convertir azimut y distancia a delta x,y en ENU:
        import math
        dx = distance * math.sin(math.radians(azimuth))  # Este
        dy = distance * math.cos(math.radians(azimuth))  # Norte

        dz = alt_ref - alt_origin  # diferencia de altura

        return dx, dy, dz


def get_waypoints(centers, num_vehicles):
    waypoints = [[] for _ in range(num_vehicles)]
    headings = [[] for _ in range(num_vehicles)]

    for center in centers:
        poses = get_formation(center, 12.0, num_vehicles)  # ENU
        ned_poses = [enu_to_ned_pose(p) for p in poses]    # Convert to NED

        for i in range(num_vehicles):
            waypoints[i].append(ned_poses[i])

    # # Calcular headings (yaw) entre cada par consecutivo de waypoints por vehículo
    # for i in range(num_vehicles):
    #     vehicle_waypoints = waypoints[i]
    #     vehicle_headings = []

    #     for j in range(len(vehicle_waypoints) - 1):
    #         p1 = vehicle_waypoints[j].position
    #         p2 = vehicle_waypoints[j + 1].position

    #         dx = p2.x - p1.x
    #         dy = p2.y - p1.y
    #         yaw = math.atan2(dy, dx)  # Heading en radianes

    #         _, _, yaw_ned = euler_enu_to_ned(yaw)
    #         vehicle_headings.append(float(yaw_ned)) 

    #     # Repetir el último heading para el último punto
    #     if vehicle_headings:
    #         vehicle_headings.append(vehicle_headings[-1])
    #     else:
    #         vehicle_headings.append(0.0)

    #     headings[i] = vehicle_headings

    return waypoints


def get_formation(center_pose: Pose, side_length: float, num_vehicles: int):

        poses = []

        if num_vehicles <= 0:
            return poses

        if num_vehicles == 1:
            poses.append(copy.deepcopy(center_pose))

        elif num_vehicles == 2:
            left = copy.deepcopy(center_pose)
            right = copy.deepcopy(center_pose)
            offset = side_length / 2.0
            left.position.x -= offset
            right.position.x += offset
            poses.extend([left, right])

        elif num_vehicles == 3:
            # Triángulo equilátero con centro en center_pose
            h = (3**0.5) / 2 * side_length  # altura del triángulo
            top = copy.deepcopy(center_pose)
            left = copy.deepcopy(center_pose)
            right = copy.deepcopy(center_pose)

            top.position.y += (2/3) * h
            left.position.y -= (1/3) * h
            left.position.x -= side_length / 2
            right.position.y -= (1/3) * h
            right.position.x += side_length / 2

            poses.extend([top, left, right])

        elif num_vehicles == 4:
            offset = side_length / 2.0

            top_left = copy.deepcopy(center_pose)
            top_left.position.x -= offset
            top_left.position.y += offset

            top_right = copy.deepcopy(center_pose)
            top_right.position.x += offset
            top_right.position.y += offset

            bottom_left = copy.deepcopy(center_pose)
            bottom_left.position.x -= offset
            bottom_left.position.y -= offset

            bottom_right = copy.deepcopy(center_pose)
            bottom_right.position.x += offset
            bottom_right.position.y -= offset

            poses.extend([top_left, top_right, bottom_left, bottom_right])

        else:
            raise ValueError("Unsupported number of vehicles: must be between 1 and 4")

        return poses

def get_initial_formation(self):
    
    target_poses = []
    target_yaws = []
    MIN_DIST_BETWEEN_DRONES = 1
    for i in self.vehicle_ids:
        pose = Pose()
        pose.position.x = self.rc_poses[i].position.x
        pose.position.y = self.rc_poses[i].position.y
        pose.position.z = -self.operation_height
        if target_poses:
            assert pose.position.x > target_poses[-1].position.x + MIN_DIST_BETWEEN_DRONES
        target_poses.append(pose)
        target_yaws.append(self.rc_yaws[i])
    return target_poses, target_yaws


def get_landing_poses(self):
        
        target_poses = []
        target_yaws = []
        for i in self.vehicle_ids:
            pose = Pose()
            pose.position.x = self.rc_poses[i].position.x
            pose.position.y = self.rc_poses[i].position.y
            pose.position.z = 0.2
            target_yaws.append(self.rc_yaws[i])
            
            target_poses.append(pose)
        return target_poses, target_yaws


def enu_ned(pose_enu):
    pose_ned = Pose()
    pose_ned.position.x = pose_enu.position.y
    pose_ned.position.y = pose_enu.position.x
    pose_ned.position.z = -pose_enu.position.z
    return pose_ned
     
def enu_ned_list(poses_enu):
    poses_ned = []
    for pose_enu in poses_enu:
        pose_ned = Pose()
        pose_ned.position.x = pose_enu.position.y
        pose_ned.position.y = pose_enu.position.x
        pose_ned.position.z = -pose_enu.position.z
        poses_ned.append(pose_ned)
    return poses_ned

def euler_enu_to_ned(yaw_enu, pitch_enu=0.0, roll_enu=0.0):

        yaw_ned = math.pi / 2 -yaw_enu

        if yaw_ned > math.pi:
            yaw_ned -= 2*math.pi 
        
        if yaw_ned < -math.pi:
            yaw_ned += 2*math.pi

        # self.get_logger().debug(f"yaw_enu 1: {yaw_enu}")

        # yaw_ned = math.pi / 2 - yaw_enu
        
        # if yaw_ned < -math.pi:
        #     yaw_ned = math.pi + (yaw_ned + math.pi)
        # if yaw_ned > math.pi:
        #     yaw_ned = -math.pi + (yaw_ned - math.pi)

        # pitch_ned = -pitch_enu
        # roll_ned = roll_enu
        pitch_ned = 0
        roll_ned = 0

        # self.get_logger().debug(f"yaw_ned 1: {yaw_ned}")
        return roll_ned, pitch_ned, yaw_ned


def enu_to_ned_pose(enu_pose: Pose) -> Pose:

        ned_pose = Pose()

        # Posición
        ned_pose.position.x = enu_pose.position.y
        ned_pose.position.y = enu_pose.position.x
        ned_pose.position.z = -enu_pose.position.z

        # Orientación: rotación 180° alrededor de X (cuaternión [1,0,0,0])
        q_enu = [
            enu_pose.orientation.w,
            enu_pose.orientation.x,
            enu_pose.orientation.y,
            enu_pose.orientation.z,
        ]

        # Cuaternión que representa la rotación 180° sobre X
        q_rot = [0, 1, 0, 0]  # w, x, y, z (en transforms3d es w,x,y,z)

        # Nota: transforms3d usa orden w,x,y,z, pero ROS usa x,y,z,w, hay que convertir
        # Primero convertir ROS (x,y,z,w) a transforms3d (w,x,y,z)
        q_enu_t3d = [enu_pose.orientation.w, enu_pose.orientation.x, enu_pose.orientation.y, enu_pose.orientation.z]

        # Rotación: q_rot * q_enu
        q_ned_t3d = tq.qmult(q_rot, q_enu_t3d)

        # Convertir de vuelta a ROS (x,y,z,w)
        ned_pose.orientation.x = q_ned_t3d[1]
        ned_pose.orientation.y = q_ned_t3d[2]
        ned_pose.orientation.z = q_ned_t3d[3]
        ned_pose.orientation.w = q_ned_t3d[0]

        return ned_pose


def enu_to_ned_twist(enu_twist: Twist) -> Twist:
    ned_twist = Twist()

    # Velocidad lineal
    ned_twist.linear.x = enu_twist.linear.y
    ned_twist.linear.y = enu_twist.linear.x
    ned_twist.linear.z = -enu_twist.linear.z

    # Velocidad angular
    ned_twist.angular.x = enu_twist.angular.y
    ned_twist.angular.y = enu_twist.angular.x
    ned_twist.angular.z = -enu_twist.angular.z

    return ned_twist
    

def convert_heading_ned_to_enu(heading_ned):
    heading_enu = heading_ned + math.pi / 2
    return heading_enu


def plot_trajectories(trajectories):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for idx, trajectory in enumerate(trajectories):
            poses = trajectory[0]

            x = [p.position.x for p in poses]
            y = [p.position.y for p in poses]
            z = [p.position.z for p in poses]  # ya está en coordenadas negativas

            ax.plot(x, y, z, label=f'Drone {idx + 1}')

            # Dibuja puntos individuales también (opcional)
            ax.scatter(x, y, z, s=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trayectorias 3D de los drones')
        ax.legend()
        ax.grid(True)
        plt.show()   


def load_points_from_csv(filename):
    poses = []
    pkg_share = get_package_share_directory('riai_planner')
    filepath = os.path.join(pkg_share, 'configuration', filename)

    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            p = Pose()
            p.position.x = float(row['x'])
            p.position.y = float(row['y'])
            p.position.z = float(row['z'])
            poses.append(p)
    return poses


def load_arucos_from_csv(filename):
    poses = {}
    pkg_share = get_package_share_directory('riai_planner')
    filepath = os.path.join(pkg_share, 'configuration', filename)

    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            p = Pose()
            p.position.x = float(row['x'])
            p.position.y = float(row['y'])
            p.position.z = float(row['z'])
            poses[row['name']] = p
    return poses


from geometry_msgs.msg import Pose
import numpy as np
import copy

def pose_to_xy(pose):
    return np.array([pose.position.x, pose.position.y])

def xy_to_pose(xy, z=0.0, orientation=None):
    p = Pose()
    p.position.x = xy[0]
    p.position.y = xy[1]
    p.position.z = z
    if orientation:
        p.orientation = copy.deepcopy(orientation)
    else:
        # orientación por defecto (sin rotación)
        p.orientation.x = 0.0
        p.orientation.y = 0.0
        p.orientation.z = 0.0
        p.orientation.w = 1.0
    return p

def barrido_zigzag_poses(corners_poses, paso=100):
    """
    corners_poses: lista de 4 Pose (ros2)
    paso: distancia entre líneas del zigzag en metros
    
    Devuelve lista de Pose con los puntos del recorrido zigzag.
    """
    # Extraer solo posiciones XY
    esquinas_xy = np.array([pose_to_xy(p) for p in corners_poses])
    
    # Ordenar esquinas: bottom-left, bottom-right, top-right, top-left
    indices = np.argsort(esquinas_xy[:,1])
    bottom = esquinas_xy[indices[:2]]
    top = esquinas_xy[indices[2:]]
    bottom = bottom[np.argsort(bottom[:,0])]
    top = top[np.argsort(top[:,0])]
    bl, br = bottom
    tl, tr = top
    
    ancho = np.linalg.norm(br - bl)
    alto = np.linalg.norm(tl - bl)
    vx = (br - bl) / ancho
    vy = (tl - bl) / alto
    
    n_lineas = int(alto // paso) + 1
    recorrido = []
    
    # Para mantener la orientación, uso la de bl
    ori = corners_poses[0].orientation
    
    for i in range(n_lineas):
        p = bl + vy * paso * i
        if i % 2 == 0:
            p1 = p
            p2 = p + vx * ancho
        else:
            p1 = p + vx * ancho
            p2 = p
        recorrido.append(xy_to_pose(p1, z=corners_poses[0].position.z, orientation=ori))
        recorrido.append(xy_to_pose(p2, z=corners_poses[0].position.z, orientation=ori))
    return recorrido



import math
from geometry_msgs.msg import Pose

def dividir_rectangulo_en_4(poses):
    """
    Recibe 4 Poses (esquinas del rectángulo en cualquier orden)
    Devuelve lista con 4 sub-rectángulos (cada uno lista de 4 Poses)
    Orden interno: [BL, BR, TR, TL]
    """
    if len(poses) != 4:
        raise ValueError("Se requieren exactamente 4 esquinas.")

    # Calcular centroide
    cx = sum(p.position.x for p in poses) / 4.0
    cy = sum(p.position.y for p in poses) / 4.0

    # Calcular ángulo desde el centro para cada punto
    def angle_from_center(p):
        return math.atan2(p.position.y - cy, p.position.x - cx)

    # Ordenar en sentido antihorario desde BL
    ordered = sorted(poses, key=angle_from_center)

    # Asignar BL, BR, TR, TL
    # Encontramos el punto más bajo para ser BL
    bl_index = min(range(4), key=lambda i: (ordered[i].position.y, ordered[i].position.x))
    ordered = ordered[bl_index:] + ordered[:bl_index]

    bl, br, tr, tl = ordered

    # Función para punto medio
    def mid(p1, p2):
        m = Pose()
        m.position.x = (p1.position.x + p2.position.x) / 2.0
        m.position.y = (p1.position.y + p2.position.y) / 2.0
        m.position.z = (p1.position.z + p2.position.z) / 2.0
        m.orientation.w = 1.0
        return m

    # Calcular puntos medios y centro
    mid_bottom = mid(bl, br)
    mid_top = mid(tl, tr)
    mid_left = mid(bl, tl)
    mid_right = mid(br, tr)
    center = mid(mid_bottom, mid_top)

    # Sub-rectángulos
    return [
        [bl, mid_bottom, center, mid_left],   # abajo izquierda
        [mid_bottom, br, mid_right, center],  # abajo derecha
        [mid_left, center, mid_top, tl],      # arriba izquierda
        [center, mid_right, tr, mid_top]      # arriba derecha
    ]


def punto_medio_pose(pose1: Pose, pose2: Pose) -> Pose:
    """Calcula el punto medio entre dos poses (solo posición, sin tocar orientación)."""
    pose_medio = Pose()
    pose_medio.position.x = (pose1.position.x + pose2.position.x) / 2.0
    pose_medio.position.y = (pose1.position.y + pose2.position.y) / 2.0
    pose_medio.position.z = (pose1.position.z + pose2.position.z) / 2.0

    # Para orientación dejamos una orientación neutra o nula, o podríamos interpolar quaternion
    # Aquí dejamos la orientación a cero
    pose_medio.orientation.x = 0.0
    pose_medio.orientation.y = 0.0
    pose_medio.orientation.z = 0.0
    pose_medio.orientation.w = 1.0

    return pose_medio

def puntos_medios_rectangulo(esquinas):
    """
    Dadas 4 poses ROS2 geometry_msgs/Pose (esquinas de un rectángulo),
    devuelve la lista de 4 poses ROS2 que son los puntos medios de los lados.
    
    Args:
        esquinas: lista de 4 objetos Pose
    
    Returns:
        lista de 4 objetos Pose con los puntos medios
    """
    puntos_medios = []
    n = len(esquinas)
    for i in range(n):
        p1 = esquinas[i]
        p2 = esquinas[(i + 1) % n]
        medio = punto_medio_pose(p1, p2)
        puntos_medios.append(medio)
    return puntos_medios


def generate_loiter_formation(center: Pose, radius: float, n_drones=3, n_points=100, speed=1.0, n_turns=1):

    if not center:
        center = Pose()
        center.pose.position.x = .0
        center.pose.position.y = .0
        center.pose.position.z = 4.0
        
    trajectories = []
    angles0 = np.linspace(0, 2 * math.pi, n_drones, endpoint=False)
    theta_vec = np.linspace(0, 2 * math.pi * n_turns, n_points * n_turns, endpoint=False)
    d_theta = theta_vec[1] - theta_vec[0]
    ang_speed = speed / radius
    dt = d_theta / ang_speed

    for theta0 in angles0:
        vels = []
        target_yaws = []
        target_dts = []

        poses = []
        for dtheta in theta_vec:
            theta = theta0 + dtheta
            p = Pose()
            p.position.x = center.position.x + radius * math.cos(theta)
            p.position.y = center.position.y + radius * math.sin(theta)
            p.position.z = center.position.z
            poses.append(p)

        for i, p in enumerate(poses):
            vel = Twist()
            if i > 0:
                prev = poses[i-1]
                dx = (p.position.x - prev.position.x)
                dy = (p.position.y - prev.position.y)
                dz = -(p.position.z - prev.position.z)
                vel.linear.x = dx / dt
                vel.linear.y = dy / dt
                vel.linear.z = dz / dt
                target_dts.append(dt)
            else:
                vel.linear.x = 0.0
                vel.linear.y = 0.0
                vel.linear.z = 0.0
                target_dts.append(0.0)
            vels.append(vel)

            dx_c = center.position.x - p.position.x
            dy_c = center.position.y - p.position.y
            yaw = math.atan2(dx_c, dy_c)
            target_yaws.append(yaw)

        trajectories.append([poses, vels, target_yaws, target_dts])

    return trajectories


def dist2(a: Pose, b: Pose) -> float:
    return math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)

def asign_circunference_points(drone_poses: list[Pose], target_poses: list[Pose]):

    dist_bd = [np.min([dist2(x, tp) for x in drone_poses]) for tp in target_poses]
    dist_bd_index = [np.argmin([dist2(x, tp) for x in drone_poses]) for tp in target_poses]
    first_target = np.argmax(dist_bd)
    first_drone = dist_bd_index[first_target]
    return [[first_drone, first_target], [1-first_drone, 1-first_target]]


def square_bounds_from_circle(pose: Pose, r: float):
    x_center = pose.position.x
    y_center = pose.position.y

    xmin = x_center - r
    xmax = x_center + r
    ymin = y_center - r
    ymax = y_center + r

    return [xmin, ymin],[xmax, ymax]

def plot_pose_list(poses: list[Pose]):
    
    xs = [p.position.x for p in poses]
    ys = [p.position.y for p in poses]
    zs = [p.position.z for p in poses]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Puntos individuales
    ax.scatter(xs, ys, zs)

    # Opcional: conectar los puntos para ver la trayectoria
    ax.plot(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trayectoria 3D de las poses')

    plt.show()