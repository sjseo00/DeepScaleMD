import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def parse_lammps_output(filepath):
    # LAMMPS thermo output: step temp etotal press
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 4:
                    data.append({
                        'step': int(parts[0]),
                        'temp': float(parts[1]),
                        'etotal': float(parts[2]),
                        'press': float(parts[3])
                    })
    return pd.DataFrame(data)

def plot_property(df1, df2, prop, label1, label2, title):
    plt.figure(figsize=(8,5))
    plt.plot(df1['step'], df1[prop], label=label1)
    plt.plot(df2['step'], df2[prop], label=label2)
    plt.xlabel('Step')
    plt.ylabel(prop)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 예시: LJ 32,000 시스템 결과 비교
    lammps_out = 'lj_32000/lammps_output.txt'  # LAMMPS output 파일 경로
    deepscalemd_out = 'lj_32000/deepscalemd_output.txt'  # DeepScaleMD output 파일 경로
    if not (os.path.exists(lammps_out) and os.path.exists(deepscalemd_out)):
        print('Output 파일을 examples/lj_32000/ 폴더에 넣어주세요.')
        return
    df_lammps = parse_lammps_output(lammps_out)
    df_deep = parse_lammps_output(deepscalemd_out)
    for prop in ['temp', 'etotal', 'press']:
        plot_property(df_lammps, df_deep, prop, 'LAMMPS', 'DeepScaleMD', f'LJ 32000 {prop} 비교')

if __name__ == '__main__':
    main()
