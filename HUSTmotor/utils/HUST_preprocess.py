import numpy as np
import scipy.io as scio
import os
from pathlib import Path


def parse_txt_file(file_path):
    print(f"正在读取文件: {file_path}")
    
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('Title') and \
               not line.startswith('Parameters') and not line.startswith('DAQ') and \
               not line.startswith('Frequency') and not line.startswith('Spectral') and \
               not line.startswith('Number') and not line.startswith('Total') and \
               not line.startswith('Channels') and not line.startswith('Legend') and \
               not line.startswith('Tacho') and not line.startswith('On/Off') and \
               not line.startswith('Volts') and not line.startswith('Time'):
                try:
                    parts = line.strip().split()
                    if len(parts) == 5:  
                        float(parts[0])  
                        data_start = i
                        break
                except:
                    continue
        
        for line in lines[data_start:]:
            line = line.strip()
            if line:
                try:
                    parts = line.split()
                    if len(parts) == 5:
                        data.append([float(x) for x in parts])
                except:
                    continue
    
    if not data:
        raise ValueError(f"无法从文件 {file_path} 中读取数据")
    
    data = np.array(data)
    print(f"  数据形状: {data.shape}")
    
    vib_data = data[:, 1:4]  # 三轴振动
    aud_data = data[:, 4]  # 声音信号
    
    return vib_data, aud_data


def create_samples(signal, sample_length, num_samples_per_class):
    total_length = len(signal)
    samples = []
    
    if num_samples_per_class > 1:
        step = (total_length - sample_length) // (num_samples_per_class - 1)
    else:
        step = 0
    
    for i in range(num_samples_per_class):
        start_idx = min(i * step, total_length - sample_length)
        sample = signal[start_idx:start_idx + sample_length]
        samples.append(sample)
    
    return np.array(samples)


def fuse_triaxial_vibration(vib_xyz):
    return np.linalg.norm(vib_xyz, axis=1)


def sample_with_start_indices(signal, start_indices, sample_length):
    return np.array([signal[s:s + sample_length] for s in start_indices])


def random_train_test_split_samples(vib_signal, aud_signal, sample_length,
                                    num_train_samples, num_test_samples, rng):
    if len(vib_signal) != len(aud_signal):
        raise ValueError("振动与声学信号长度不一致")

    total_needed = num_train_samples + num_test_samples
    max_start = len(vib_signal) - sample_length + 1

    if max_start <= 0:
        raise ValueError("信号长度小于样本长度，无法切片")
    if total_needed > max_start:
        raise ValueError(
            f"样本需求({total_needed})超过可用不重复起点数({max_start})"
        )

    start_indices = rng.choice(max_start, size=total_needed, replace=False)
    rng.shuffle(start_indices)

    train_starts = start_indices[:num_train_samples]
    test_starts = start_indices[num_train_samples:]

    vib_train = sample_with_start_indices(vib_signal, train_starts, sample_length)
    vib_test = sample_with_start_indices(vib_signal, test_starts, sample_length)
    aud_train = sample_with_start_indices(aud_signal, train_starts, sample_length)
    aud_test = sample_with_start_indices(aud_signal, test_starts, sample_length)

    return vib_train, vib_test, aud_train, aud_test


def convert_txt_to_mat(txt_folder, vib_output_path, aud_output_path,
                       vib_sample_length=1024, 
                       aud_sample_length=1024,
                       num_train_samples=800,
                       num_test_samples=200,
                       random_seed=42):

    if vib_sample_length != aud_sample_length:
        raise ValueError("当前实现要求振动和声学样本长度一致")
    
    # 健康状态映射 (6种状态对应类别0-5)
    health_states = ['H', 'BF', 'BOW', 'BROKEN', 'MISAL', 'UNBAL']
    # 工况映射 (4种工况对应domain 1-4)
    working_conditions = ['5HZ', '10HZ', '20HZ', '30HZ']
    
    # 创建两个独立的存储字典
    vib_mat_data = {}
    aud_mat_data = {}
    rng = np.random.default_rng(random_seed)
    
    # 处理每个工况
    for domain_idx, wc in enumerate(working_conditions, start=1):
        print(f"\n{'='*60}")
        print(f"处理工况 {domain_idx}: {wc}")
        print(f"{'='*60}")
        
        train_vib_list = []
        train_aud_list = []
        test_vib_list = []
        test_aud_list = []
        
        for class_idx, health_state in enumerate(health_states):
            filename = f"{health_state}_{wc}.txt"
            file_path = os.path.join(txt_folder, filename)
            
            if not os.path.exists(file_path):
                print(f"  ⚠️  警告: 文件不存在 {filename}，跳过")
                continue
            
            print(f"  处理类别 {class_idx + 1}/{len(health_states)}: {health_state}")
            
            vib_xyz, aud_signal = parse_txt_file(file_path)
            vib_signal = fuse_triaxial_vibration(vib_xyz)

            vib_train, vib_test, aud_train, aud_test = random_train_test_split_samples(
                vib_signal=vib_signal,
                aud_signal=aud_signal,
                sample_length=vib_sample_length,
                num_train_samples=num_train_samples,
                num_test_samples=num_test_samples,
                rng=rng
            )
            
            train_vib_list.append(vib_train)
            train_aud_list.append(aud_train)
            test_vib_list.append(vib_test)
            test_aud_list.append(aud_test)
            
            print(f"    ✓ 训练样本: 振动 {vib_train.shape}, 声学 {aud_train.shape}")
            print(f"    ✓ 测试样本: 振动 {vib_test.shape}, 声学 {aud_test.shape}")
        
        # 合并所有类别的数据并存储
        if train_vib_list:
            # 振动数据: shape = (sample_length, num_classes * num_samples)
            vib_mat_data[f'load{domain_idx}_train'] = np.vstack(train_vib_list).T
            vib_mat_data[f'load{domain_idx}_test'] = np.vstack(test_vib_list).T
            
            # 声学数据: shape = (sample_length, num_classes * num_samples)
            aud_mat_data[f'load{domain_idx}_train'] = np.vstack(train_aud_list).T
            aud_mat_data[f'load{domain_idx}_test'] = np.vstack(test_aud_list).T
            
            print(f"\n  📊 工况{domain_idx}汇总:")
            print(f"    振动数据 - 训练: {vib_mat_data[f'load{domain_idx}_train'].shape}, "
                  f"测试: {vib_mat_data[f'load{domain_idx}_test'].shape}")
            print(f"    声学数据 - 训练: {aud_mat_data[f'load{domain_idx}_train'].shape}, "
                  f"测试: {aud_mat_data[f'load{domain_idx}_test'].shape}")
    
    # 保存振动数据MAT文件
    print(f"\n{'='*60}")
    print(f"💾 保存振动数据到: {vib_output_path}")
    print(f"{'='*60}")
    scio.savemat(vib_output_path, vib_mat_data)
    print("✓ 振动数据保存成功！")
    
    # 保存声学数据MAT文件
    print(f"\n{'='*60}")
    print(f"💾 保存声学数据到: {aud_output_path}")
    print(f"{'='*60}")
    scio.savemat(aud_output_path, aud_mat_data)
    print("✓ 声学数据保存成功！")
    
    print(f"\n{'='*60}")
    print("🎉 转换完成！")
    print(f"{'='*60}")
    print(f"振动数据文件: {vib_output_path}")
    print(f"声学数据文件: {aud_output_path}")
    
    return vib_output_path, aud_output_path


def verify_mat_file(mat_path):
    """验证生成的.mat文件"""
    print(f"\n验证文件: {mat_path}")
    data = scio.loadmat(mat_path)
    
    print("文件中的变量:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {data[key].shape}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    txt_folder = str(data_dir)  # txt文件所在文件夹
    vib_output = str(data_dir / "Motor_Vib.mat")  # 振动数据输出文件
    aud_output = str(data_dir / "Motor_Aud.mat")  # 声学数据输出文件
    
    # - 振动样本长度: 1024
    # - 声学样本长度: 1024  
    # - 训练样本: 800/类
    # - 测试样本: 200/类
    
    print("=" * 60)
    print("HUSTmotor 数据集 TXT 转 MAT 转换器")
    print("=" * 60)
    print(f"\n输入文件夹: {txt_folder}")
    print(f"输出文件:")
    print(f"  - 振动数据: {vib_output}")
    print(f"  - 声学数据: {aud_output}")
    
    vib_path, aud_path = convert_txt_to_mat(
        txt_folder=txt_folder,
        vib_output_path=vib_output,
        aud_output_path=aud_output,
        vib_sample_length=1024,
        aud_sample_length=1024,
        num_train_samples=800,
        num_test_samples=200,
        random_seed=42
    )
    
    print(f"\n{'='*60}")
    print("📋 文件验证")
    print(f"{'='*60}")
    verify_mat_file(vib_path)
    verify_mat_file(aud_path)
    
    print("\n" + "=" * 60)
    print("✅ 转换成功完成!")
    print("=" * 60)
    print("\n📝 下一步:")
    print("  1. 确认 TXT 文件已经放在 HUSTmotor/data/ 下")
    print("  2. 训练脚本会默认读取以下文件:")
    print(f"     root_vib = '{vib_output}'")
    print(f"     root_asc = '{aud_output}'")
