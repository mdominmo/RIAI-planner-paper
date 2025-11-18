#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ruta al archivo
file_path = "/home/manuel/Documents/jjaa_experiments.ods"

# --- Carga y preprocesamiento de datos ---
try:
    raw_df = pd.read_excel(file_path, engine="odf", header=None, skiprows=5)
    df = raw_df.iloc[:, 7:14].copy()
    df.columns = ['experimento', 'pos_inicial', 'asignacion', 'tiempo (s)', 'num_drones', 'num_arucos', 'caso']
    df['tiempo (s)'] = pd.to_numeric(df['tiempo (s)'], errors='coerce')
    df = df.dropna(subset=['tiempo (s)'])

    caso_order = ['m<n', 'm=n', 'm>n']
    df['caso'] = df['caso'].astype(str).str.strip()
    df['caso'] = pd.Categorical(df['caso'], categories=caso_order, ordered=True)

except FileNotFoundError:
    print(f"Error: El archivo no se encuentra en la ruta especificada: {file_path}")
    exit()
except Exception as e:
    print(f"Ocurrió un error durante la carga o preprocesamiento del archivo: {e}")
    exit()

# --- Estilo de barra ---
def get_bar_style(row, color_map):
    base_color = color_map.get(row['caso'], 'gray')
    alpha = 1.0 if row['asignacion'] != 'random' else 0.6
    return base_color, alpha

# --- Función principal ---
def plot_experimentos_individuales_por_caso(pos):
    subset = df[df['pos_inicial'] == pos].copy()
    if subset.empty:
        print(f"No hay datos para la posición inicial: {pos}")
        return

    subset_sorted = subset.sort_values(by=['caso', 'experimento', 'asignacion'])
    subset_sorted['x_pos_individual_bar'] = np.arange(len(subset_sorted))

    base_color_map = {
        'm=n': 'red',
        'm<n': 'green',
        'm>n': 'blue'
    }

    subset_sorted[['bar_color', 'bar_alpha']] = subset_sorted.apply(
        lambda row: get_bar_style(row, base_color_map), axis=1, result_type='expand'
    )

    x_positions = subset_sorted['x_pos_individual_bar'].values
    bar_heights = subset_sorted['tiempo (s)'].values
    bar_colors = subset_sorted['bar_color'].values
    bar_alphas = subset_sorted['bar_alpha'].values

    bar_width = 0.6
    fig, ax = plt.subplots(figsize=(max(16, len(subset_sorted) * 0.4), 10))

    bars = ax.bar(x_positions, bar_heights, width=bar_width, color=bar_colors)
    for i, bar in enumerate(bars.patches):
        bar.set_alpha(bar_alphas[i])

    # Etiquetas de ticks
    tick_labels = (subset_sorted['x_pos_individual_bar'] + 1).astype(str).tolist()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=36, rotation=90, ha='center', va='top')
    ax.tick_params(axis='x', pad=20, labelsize=36)

    ax.set_ylabel('Tiempo (s)', fontsize=40, labelpad=30)
    ax.set_xlabel('Número de experimento', fontsize=40, labelpad=30)
    ax.tick_params(axis='y', labelsize=36)

    ax.set_title('Tiempo por experimento y asignación', fontsize=44, pad=40)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Líneas divisorias entre casos
    case_change_x_pos = subset_sorted.groupby('caso').head(1)['x_pos_individual_bar'].tolist()
    for i in range(1, len(case_change_x_pos)):
        sep_pos = case_change_x_pos[i] - 0.5
        if sep_pos > x_positions[0] - 0.5 and sep_pos < x_positions[-1] + 0.5:
            ax.axvline(x=sep_pos, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

    # Leyenda en figura aparte con 2 columnas
    legend_handles = []
    legend_labels = []
    for caso in caso_order:
        base_color = base_color_map.get(caso, 'gray')
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=base_color, alpha=1.0))
        legend_labels.append(f'{caso} (Asignación óptima)')
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=base_color, alpha=0.6))
        legend_labels.append(f'{caso} (Asignación aleatoria)')

    fig_legend = plt.figure(figsize=(14, 4))  # Altura más generosa
    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        fontsize=18,
        ncol=2,  # 2 elementos por fila
        frameon=False
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Ejecutar
plot_experimentos_individuales_por_caso('esquinas')
