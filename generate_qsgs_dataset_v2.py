import numpy as np
import cv2
import os
import time
from scipy.ndimage import label
import logging

# 设置日志配置，使用DEBUG级别以获取更详细的信息
logging.basicConfig(filename='qsgs_generation.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_qsgs(width, height, target_porosity, cd_probability):
    """
    使用QSGS算法生成2D多孔介质微观结构
    
    参数:
    width, height: 图像尺寸
    target_porosity: 目标孔隙率 (孔隙空间占比)
    cd_probability: 种子概率
    
    返回:
    image_array: 生成的图像数组，0表示孔隙，1表示固体
    """
    # 初始化网格为0 (全部孔隙)
    grid = np.zeros((height, width), dtype=np.uint8)
    
    # 计算目标固体像素数量
    total_pixels = width * height
    target_solid = int((1 - target_porosity) * total_pixels)
    
    logging.debug(f"QSGS Generation: target_porosity={target_porosity}, cd_probability={cd_probability}")
    logging.debug(f"QSGS Generation: total_pixels={total_pixels}, target_solid={target_solid}")
    
    # 相位1: 播种
    # 随机生成种子 (值为1表示固体种子)
    seeds = np.random.rand(height, width) < cd_probability
    grid[seeds] = 1
    initial_solid = np.sum(grid)
    logging.debug(f"QSGS Generation: initial_solid={initial_solid}, initial_porosity={1 - (initial_solid / total_pixels):.4f}")
    
    # 相位2: 生长
    # 8邻域方向 (包括对角线)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    current_solid = initial_solid
    iterations = 0
    max_iterations = total_pixels * 2  # 防止无限循环
    
    # 使用集合来高效跟踪可生长的位置，避免重复计算
    # 初始时，收集所有固体像素的空邻域
    growth_candidates = set()
    
    # 收集初始种子的所有空邻域
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 1:
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == 0:
                        growth_candidates.add((ny, nx))
    
    logging.debug(f"QSGS Generation: initial growth_candidates={len(growth_candidates)}")
    
    # 生长主循环
    while current_solid < target_solid and iterations < max_iterations and growth_candidates:
        iterations += 1
        
        # 定期记录生长进度
        if iterations % 1000 == 0:
            logging.debug(f"QSGS Generation: iteration={iterations}, current_solid={current_solid}, growth_candidates={len(growth_candidates)}")
        
        # 随机选择一个生长位置
        growth_list = list(growth_candidates)
        idx = np.random.randint(len(growth_list))
        ny, nx = growth_list[idx]
        
        # 生长：将孔隙变为固体
        grid[ny, nx] = 1
        current_solid += 1
        
        # 从候选列表中移除当前生长位置
        growth_candidates.remove((ny, nx))
        
        # 添加新的生长候选位置：检查新生长像素的所有空邻域
        for dy, dx in neighbors:
            new_y, new_x = ny + dy, nx + dx
            if 0 <= new_y < height and 0 <= new_x < width and grid[new_y, new_x] == 0:
                growth_candidates.add((new_y, new_x))
    
    logging.debug(f"QSGS Generation: growth_iterations={iterations}, current_solid={current_solid}")
    
    # 调整到目标孔隙率 (如果需要)
    current_porosity = 1 - (current_solid / total_pixels)
    logging.debug(f"QSGS Generation: before adjustment - current_porosity={current_porosity:.4f}")
    
    if abs(current_porosity - target_porosity) > 0.005:
        # 微调：如果固体太多，随机将一些固体转为孔隙
        if current_porosity < target_porosity:
            solid_positions = np.argwhere(grid == 1)
            excess = current_solid - target_solid
            if excess > 0:
                to_remove = np.random.choice(len(solid_positions), min(excess, len(solid_positions)), replace=False)
                for idx in to_remove:
                    y, x = solid_positions[idx]
                    grid[y, x] = 0
                current_solid -= len(to_remove)
        # 如果固体太少，随机将一些孔隙转为固体
        else:
            pore_positions = np.argwhere(grid == 0)
            deficit = target_solid - current_solid
            if deficit > 0:
                to_add = np.random.choice(len(pore_positions), min(deficit, len(pore_positions)), replace=False)
                for idx in to_add:
                    y, x = pore_positions[idx]
                    grid[y, x] = 1
                current_solid += len(to_add)
    
    final_porosity = 1 - (current_solid / total_pixels)
    logging.debug(f"QSGS Generation: after adjustment - final_porosity={final_porosity:.4f}")
    
    return grid

def smooth_adjust_by_threshold(image, target_porosity, sigma=0.5):
    """
    使用高斯模糊+动态阈值法调整孔隙率。
    既能保证边缘绝对光滑（无毛刺），又能保证孔隙率精准。
    
    参数:
    image: 输入的二值图像 (0/1)
    target_porosity: 目标孔隙率
    sigma: 平滑程度。值越大越圆润，但可能会吞掉极小的细节。推荐 0.5 到 1.5。
    """
    # 1. 转换为浮点数
    img_float = image.astype(np.float32)
    
    # 2. 应用高斯模糊，把二值图变成连续的灰度图
    # 这步操作会自动把"毛刺"磨平，变成平滑的梯度
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 3. 寻找精确的截断阈值
    # 我们需要找到一个阈值 T，使得大于 T 的像素数量刚好等于 target_solid
    total_pixels = image.size
    target_solid_count = int(total_pixels * (1 - target_porosity))
    
    # 将模糊后的图像展平并从大到小排序
    flat_pixels = np.sort(blurred.ravel())[::-1]
    
    # 取第 target_solid_count 个像素的值作为阈值
    # 稍微加一点极小值epsilon，确保边界处理稳定
    if target_solid_count >= total_pixels:
        threshold = -1.0 # 全是固体
    elif target_solid_count <= 0:
        threshold = 2.0 # 全是孔隙
    else:
        threshold = flat_pixels[target_solid_count - 1]
    
    # 4. 根据阈值重新二值化
    new_grid = np.zeros_like(image)
    new_grid[blurred >= threshold] = 1
    
    # 5. 极微小的修正 (处理阈值恰好等于多个像素值的情况)
    # 因为浮点数精度问题，直接截断可能会有几个像素的误差，这里做最后一次强制修正
    current_solid = np.sum(new_grid)
    diff = target_solid_count - current_solid
    
    if diff != 0:
        # 找出刚好等于阈值的边缘像素
        edge_mask = (blurred == threshold)
        edge_coords = np.argwhere(edge_mask)
        
        if len(edge_coords) > 0:
            if diff > 0: # 少了，把一些边缘变成1
                # 优先选原本就是1的像素（如果有的话），或者随机选
                # 简单起见，随机选
                indices = np.random.choice(len(edge_coords), min(diff, len(edge_coords)), replace=False)
                for idx in indices:
                    new_grid[edge_coords[idx][0], edge_coords[idx][1]] = 1
            elif diff < 0: # 多了，把一些边缘变成0
                indices = np.random.choice(len(edge_coords), min(abs(diff), len(edge_coords)), replace=False)
                for idx in indices:
                    new_grid[edge_coords[idx][0], edge_coords[idx][1]] = 0

    return new_grid

    
def check_percolation(image_array):
    """
    检查孔隙空间是否从左边界渗流到右边界
    
    参数:
    image_array: 图像数组，0表示孔隙，1表示固体
    
    返回:
    bool: 如果孔隙渗流，返回True，否则返回False
    """
    # 创建孔隙掩码：1表示孔隙，0表示固体
    pore_mask = (image_array == 0).astype(int)
    
    # 标记连通区域
    labeled_array, num_features = label(pore_mask)
    logging.debug(f"Percolation Check: num_features={num_features}")
    
    if num_features == 0:
        logging.debug(f"Percolation Check: No pore space found")
        return False
    
    # 获取左边界和右边界的连通区域标签
    left_boundary = labeled_array[:, 0]
    right_boundary = labeled_array[:, -1]
    
    left_labels = set(left_boundary)
    right_labels = set(right_boundary)
    
    # 排除固体区域 (标签为0)
    left_labels.discard(0)
    right_labels.discard(0)
    
    logging.debug(f"Percolation Check: left_labels={left_labels}, right_labels={right_labels}")
    
    # 检查是否有共同的标签
    common_labels = left_labels & right_labels
    logging.debug(f"Percolation Check: common_labels={common_labels}, is_percolating={bool(common_labels)}")
    
    return bool(common_labels)

def main():
    logging.info("Starting QSGS dataset generation")
    
    # 设置参数
    image_size = 128
    porosities = np.arange(0.40, 0.85, 0.05)  # 0.40到0.80，步长0.05
    # 大幅降低种子概率，防止椒盐噪声
    modes = {
        'A': 0.01,   # 精细：极低的种子概率
        'B': 0.005,  # 中等：更低的种子概率
        'C': 0.0005  # 粗糙：非常低的种子概率，生成大簇
    }
    
    # 计算每个组合需要生成的图像数量，以达到总约10000张图像
    total_combinations = len(porosities) * len(modes)
    images_per_combination = 100000 // total_combinations
    total_target_images = total_combinations * images_per_combination
    logging.info(f"Parameters: width={image_size}, height={image_size}")
    logging.info(f"Porosities: {porosities}")
    logging.info(f"Modes: {modes}")
    logging.info(f"Images per combination: {images_per_combination}")
    logging.info(f"Total images to generate: {total_target_images}")
    
    output_dir = 'data/raw_images'
    logging.info(f"Output directory: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # 主循环初始化
    total_images = 0
    start_time = time.time()
    combination_index = 0
    
    # 生成所有组合的列表，用于进度计算
    all_combinations = [(porosity, mode, cd_prob) for porosity in porosities for mode, cd_prob in modes.items()]
    total_combinations = len(all_combinations)
    
    print("=" * 80)
    print("QSGS 微观结构数据集生成")
    print("=" * 80)
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"孔隙率范围: {min(porosities):.2f} - {max(porosities):.2f} (步长: 0.05)")
    print(f"生成模式: {list(modes.keys())}")
    print(f"目标总图像数: {total_target_images}")
    print(f"每个组合生成: {images_per_combination} 张")
    print("=" * 80)
    print()
    
    # 主循环
    for porosity, mode, cd_prob in all_combinations:
        combination_index += 1
        count = 0
        attempts = 0
        max_attempts = images_per_combination * 2  # 尝试次数为目标数量的2倍
        
        # 组合进度信息
        print(f"[{combination_index}/{total_combinations}] 正在生成: 孔隙率={porosity:.2f}, 模式={mode}")
        print(f"种子概率: {cd_prob:.6f}, 目标图像数: {images_per_combination}")
        
        combination_start_time = time.time()
        
        while count < images_per_combination and attempts < max_attempts:
            attempts += 1
            # 1. 生成原始粗糙的 QSGS
            image_array = generate_qsgs(image_size, image_size, porosity, cd_prob)
            
            # 2. [替换旧逻辑] 使用高斯阈值法，一步到位实现“平滑+精准控孔”
            # sigma=1.0 是一个平衡点，如果你想要更圆滚滚的效果，可以设为 1.5 或 2.0
            image_array = smooth_adjust_by_threshold(image_array, porosity, sigma=1.0)
            
            # 计算实际孔隙率
            actual_porosity = np.sum(image_array == 0) / image_array.size
            
            # 3. 检查渗流
            is_connected = check_percolation(image_array)
            suffix = "conn" if is_connected else "block"
            
            # 4. 保存...
            png_array = (image_array * 255).astype(np.uint8)
            
            # 保存图像
            filename = f"{porosity:.2f}_{mode}_{count+371:03d}_{suffix}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, png_array)
            
            count += 1
            total_images += 1
            
            # 实时进度输出
            if count % 5 == 0 or count == images_per_combination:
                # 计算当前组合的进度
                combination_progress = count / images_per_combination * 100
                # 计算整体进度
                overall_progress = total_images / total_target_images * 100
                
                # 计算已用时间和估计剩余时间
                elapsed_time = time.time() - start_time
                if total_images > 0:
                    avg_time_per_image = elapsed_time / total_images
                    estimated_remaining = (total_target_images - total_images) * avg_time_per_image
                    remaining_str = f"剩余时间: {int(estimated_remaining // 60)}分{int(estimated_remaining % 60)}秒"
                else:
                    remaining_str = "剩余时间: 计算中..."
                
                # 进度条显示
                bar_length = 50
                overall_bar = '█' * int(overall_progress / 100 * bar_length) + '-' * (bar_length - int(overall_progress / 100 * bar_length))
                combination_bar = '█' * int(combination_progress / 100 * bar_length) + '-' * (bar_length - int(combination_progress / 100 * bar_length))
                
                # 清除当前行及以下的输出，解决视觉暂留问题
                print("\x1b[2K", end="")  # 清除当前行
                print(f"组合进度: [{combination_bar}] {combination_progress:.1f}%")
                print("\x1b[2K", end="")  # 清除当前行
                print(f"整体进度: [{overall_bar}] {overall_progress:.1f}%")
                print("\x1b[2K", end="")  # 清除当前行
                print(f"已生成: {count}/{images_per_combination} 张 | 总图像: {total_images}/{total_target_images} 张 | 实际孔隙率: {actual_porosity:.4f} | {remaining_str}", end="\r")
            
            logging.info(f"Saved {filename}, target porosity: {porosity:.2f}, actual porosity: {actual_porosity:.4f}, mode: {mode}, count: {count}/{images_per_combination}")
        
        combination_elapsed = time.time() - combination_start_time
        print()
        print(f"✓ 完成: 孔隙率={porosity:.2f}, 模式={mode}")
        print(f"生成图像: {count}/{images_per_combination} 张 | 耗时: {int(combination_elapsed // 60)}分{int(combination_elapsed % 60)}秒")
        print()
        
        if attempts >= max_attempts:
            logging.warning(f"Max attempts reached for porosity={porosity:.2f}, mode={mode}. Generated {count}/{images_per_combination} images.")
    
    # 完成信息
    total_elapsed = time.time() - start_time
    print("=" * 80)
    print("🎉 QSGS 微观结构数据集生成完成!")
    print(f"总生成图像数: {total_images}")
    print(f"总耗时: {int(total_elapsed // 60)}分{int(total_elapsed % 60)}秒")
    print(f"平均生成速度: {total_images / total_elapsed:.2f} 张/秒")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    logging.info(f"QSGS dataset generation completed! Total images generated: {total_images}")
    logging.info(f"Total elapsed time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()