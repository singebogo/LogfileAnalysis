import mmap
import yappi
import os
import subprocess
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd
import plotly.express as px
from flask import render_template
from jinja2 import filters
from functools import partial
from typing import Dict, Any, List, Callable, Union, Tuple
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
from flask import copy_current_request_context
from .servers import *


########################################################################
# 增加功能
## 1、增加选择需要解析的文件目录
## 2、日志文件中所有的存储数据
## 3、数据库配置，配置化
#########
#########
#########
#########
########################################################################

# 配置参数

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """高亮显示关键词"""
    for keyword in keywords:
        text = re.sub(
            f'({keyword})',
            r'<span class="keyword-highlight">\1</span>',
            text,
            flags=re.IGNORECASE
        )
    return text


def get_buffer_size(file_path: str) -> int:
    """动态计算最佳缓冲区大小"""
    file_size = os.path.getsize(file_path)
    return min(max(file_size // 100, 64 * 1024), 8 * 1024 * 1024)


def get_analyzer(file_path: str, user_id) -> Callable[[str], Dict[str, Any]]:
    """根据文件大小选择分析方法"""
    return (analyze_with_mmap
            if os.path.getsize(file_path) > getConfig(user_id, 'performance', 'min_mmap_size')
            else analyze_with_buffer)


def byte_size(user_id):
    try:
        buffer_size = int(getConfig(user_id, 'performance', 'byte_cache_size'))
        buffer_size = max(10 * 1024 * 1024, min(buffer_size, 64 * 1024 * 1024))  # 限制在64KB~64MB之间
    except (ValueError, KeyError):
        buffer_size = 10 * 1024 * 1024  # 默认6MB
    return buffer_size


def count_lines_optimized(file_path: str, user_id: str) -> Union[int, Tuple[int, Any]]:
    """
    安全优化的行数统计实现
    参数:
        file_path: 待统计的文件路径
        user_id: 用于获取用户特定配置
    返回:
        (行数, 使用的缓冲区大小)
    """

    buffer_size = byte_size(user_id)
    line_count = 0
    last_byte = b'\n'

    try:
        with open(file_path, 'rb') as f:
            file_size = os.path.getsize(file_path)

            # 小文件直接处理
            if file_size <= 2 * buffer_size:
                data = f.read()
                line_count = data.count(b'\n')
                last_byte = data[-1:] if data else b'\n'
            # 大文件处理
            else:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                    # 在mmap上下文中完成所有memoryview操作
                    # 处理所有chunk
                    for offset in range(0, file_size, buffer_size):
                        chunk_size = min(buffer_size, file_size - offset)
                        chunk = m[offset:offset + chunk_size]
                        line_count += chunk.count(b'\n')

                    # 获取最后字节（必须在mmap上下文中完成）
                    if m.size() > 0:
                        last_byte = m[-1:]
        # 处理最后一行（此时mmap已安全关闭）
        if last_byte != b'\n':
            line_count += 1

    except PermissionError as pe:
        logger.error(f"Permission denied: {file_path}")
        raise
    except OSError as oe:
        logger.error(f"File access error: {str(oe)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

    return line_count, buffer_size


def count_lines(file_path: str, user_id) -> Union[int, tuple[int, Any]]:
    """跨平台的行数统计实现"""
    # 根据系统选择最优方案
    if os.name == 'posix':  # Unix/Linux/Mac
        try:
            result = subprocess.run(['wc', '-l', file_path],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return int(result.stdout.split()[0]), byte_size(user_id)
        except:
            pass  # 失败时回退到缓冲计数

    return count_lines_optimized(file_path, user_id)


def init_analyze_files(file_path: str, user_id, batch_id):
    # Get total lines first for progress tracking
    total_lines, byte_size = count_lines(file_path, user_id)

    # Create analyzed file record
    analyzed_file_id = create_analyzed_file(file_path, user_id, total_lines, batch_id)
    return analyzed_file_id, total_lines, byte_size


def analyze_with_buffer(file_path: str, user_id, batch_id) -> Dict[str, Any]:
    """使用缓冲读取分析文件"""
    analyzed_file_id, total_lines, byte_size = init_analyze_files(file_path, user_id, batch_id)
    buffer_size = get_buffer_size(file_path)
    results = init_results(file_path)
    logging.info("开始解析：{}".format(file_path))

    TIME_REGEX = time_regex(user_id)
    KEYWORD_REGEXES = kwds_regex(user_id)
    EXCEPTION_REGEX = exceptions_regex(user_id)
    EXCEPTION_PATTERNS = getConfig(user_id, 'exception', 'patterns')

    # 优化点1：批量处理行以减少函数调用开销
    batch_size = getConfig(user_id, 'performance', 'batch_size')  # 每次处理5000 行
    line_batch = []
    last_update = 0

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace',
                  buffering=buffer_size) as f:
            # for line_number, line in enumerate(f, 1):
            #     process_line(line, line_number, results, user_id, TIME_REGEX, KEYWORD_REGEXES, EXCEPTION_REGEX, EXCEPTION_PATTERNS)
            #
            #     # Update progress every 100 lines or at the end
            #     if line_number % 5000 == 0 or line_number == total_lines:
            #         update_analysis_progress(analyzed_file_id, line_number, total_lines)
            # 优化点2：使用更快的行迭代方式
            while True:
                lines = f.readlines(byte_size)
                if not lines:
                    break

                for line_number, line in enumerate(lines, last_update + 1):
                    line = line.rstrip('\n')
                    line_batch.append((line, line_number))

                    # 批量处理
                    if len(line_batch) >= batch_size:
                        process_lines_batch(
                            line_batch,
                            results,
                            user_id,
                            TIME_REGEX,
                            KEYWORD_REGEXES,
                            EXCEPTION_REGEX,
                            EXCEPTION_PATTERNS
                        )
                        line_batch = []

                        # 优化点3：减少进度更新频率
                        if line_number - last_update >= batch_size or line_number == total_lines:
                            update_analysis_progress(analyzed_file_id, line_number, total_lines)
                            last_update = line_number

                # 处理剩余行
                if line_batch:
                    process_lines_batch(
                        line_batch,
                        results,
                        user_id,
                        TIME_REGEX,
                        KEYWORD_REGEXES,
                        EXCEPTION_REGEX,
                        EXCEPTION_PATTERNS
                    )
        # 生成单个文件的时间分布图
        generate_file_plots(results)

        # Mark file as completed
        mark_analysis_complete(analyzed_file_id)

    except Exception as e:
        logger.error(f"分析文件 analyze_with_buffer {file_path} 时出错: {str(e)}")
        mark_analysis_failed(analyzed_file_id, str(e))

    return results


def analyze_with_mmap(file_path: str, user_id, batch_id) -> Dict[str, Any]:
    """优化后的内存映射文件分析函数（使用批量处理）"""
    analyzed_file_id, total_lines, byte_size = init_analyze_files(file_path, user_id, batch_id)
    results = init_results(file_path)

    # 预编译正则表达式
    TIME_REGEX = time_regex(user_id)
    KEYWORD_REGEXES = kwds_regex(user_id)
    EXCEPTION_REGEX = exceptions_regex(user_id)
    EXCEPTION_PATTERNS = getConfig(user_id, 'exception', 'patterns')

    # 优化参数
    BATCH_SIZE = getConfig(user_id, 'performance', 'batch_size')  # 每次处理5000 行
    UPDATE_INTERVAL = BATCH_SIZE  # 进度更新间隔
    DECODE_BUFFER = byte_size  # 1MB解码缓冲区

    try:
        with open(file_path, 'rb') as f:  # 使用二进制模式打开
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                line_number = 1
                start = 0
                last_update = 0
                decode_buffer = bytearray()
                line_batch = []

                while True:
                    end = mm.find(b'\n', start)
                    if end == -1:
                        chunk = mm[start:]
                        if not chunk:
                            break
                    else:
                        chunk = mm[start:end + 1]  # 包含换行符

                    # 添加到解码缓冲区
                    decode_buffer.extend(chunk)

                    # 当缓冲区足够大时批量解码
                    if len(decode_buffer) >= DECODE_BUFFER or end == -1:
                        try:
                            # 批量解码缓冲区内容
                            text_chunk = decode_buffer.decode('utf-8', errors='replace')
                            lines = text_chunk.splitlines()

                            # 准备批量处理
                            batch_lines = [(line, line_number + i) for i, line in enumerate(lines) if line]
                            line_batch.extend(batch_lines)
                            line_number += len(lines)

                            # 当达到批次大小时处理
                            if len(line_batch) >= BATCH_SIZE or end == -1:
                                process_lines_batch(
                                    line_batch,
                                    results,
                                    user_id,
                                    TIME_REGEX,
                                    KEYWORD_REGEXES,
                                    EXCEPTION_REGEX,
                                    EXCEPTION_PATTERNS
                                )
                                line_batch = []

                            # 更新进度
                            if line_number - last_update >= UPDATE_INTERVAL or line_number == total_lines:
                                update_analysis_progress(analyzed_file_id, line_number, total_lines)
                                last_update = line_number

                        except UnicodeDecodeError:
                            # 处理解码错误，逐行处理
                            for line in decode_buffer.splitlines():
                                try:
                                    line_text = line.decode('utf-8', errors='replace')
                                    if line_text:
                                        process_line(
                                            line_text, line_number, results,
                                            user_id, TIME_REGEX, KEYWORD_REGEXES,
                                            EXCEPTION_REGEX, EXCEPTION_PATTERNS
                                        )
                                        line_number += 1
                                except Exception:
                                    continue

                        finally:
                            decode_buffer.clear()  # 清空缓冲区

                    if end == -1:
                        break
                    start = end + 1

        # 生成图表并标记完成
        generate_file_plots(results)
        mark_analysis_complete(analyzed_file_id)

    except Exception as e:
        logger.error(f"分析文件 {file_path} 时出错: {str(e)}", exc_info=True)
        mark_analysis_failed(analyzed_file_id, str(e))
        raise

    return results


def process_lines_batch(lines_batch, results, user_id, TIME_REGEX, KEYWORD_REGEXES, EXCEPTION_REGEX,
                        EXCEPTION_PATTERNS):
    """
      优化后的批量行处理函数
      参数:
          lines_batch: 包含(行内容, 行号)的元组列表
          results: 结果字典
          user_id: 用户ID
          TIME_REGEX: 预编译的时间正则
          KEYWORD_REGEXES: 预编译的关键词正则字典
          EXCEPTION_REGEX: 预编译的异常正则
          EXCEPTION_PATTERNS: 异常模式配置
    """
    user_keywords = kwrods(user_id)  # 预先获取

    for line, line_number in lines_batch:
        line = clean_text(line)
        line = line.rstrip('\r')  # 更快的行清理
        results['stats']['lines'] += 1

        # 使用预编译的正则表达式检查关键词
        found_keywords = []
        for kw, regex in KEYWORD_REGEXES.items():
            if regex.search(line):
                found_keywords.append(kw)
                results['keyword_counts'][kw] += 1
                results['all_keywords'].add(kw)

        if not found_keywords:
            continue

        if found_keywords:
            results['stats']['keywords'] += len(found_keywords)
            match = TIME_REGEX.search(line)
            error_time = match.group(1) if match else '未知时间'

            for keyword in found_keywords:
                detail = {
                    'source': results['filename'],
                    'line_number': line_number,
                    'time': error_time,
                    'keyword': keyword,
                    'message': line[:500].strip(),  # 限制消息长度
                }

                if keyword in user_keywords:
                    results['error_details'].append(detail)
                    results['stats']['errors'] += 1
                    if match:
                        try:
                            error_time = datetime.strptime(match.group(1), '%Y-%m-%d%H:%M:%S.%f')
                            results['error_times'].append(error_time)
                        except ValueError:
                            pass
                else:
                    exception_match = EXCEPTION_REGEX.search(line)
                    exception_type = exception_match.group() if exception_match else EXCEPTION_PATTERNS['OtherError']
                    detail['exception'] = exception_type
                    results['exception_counts'][exception_type] += 1
                    results['all_exceptions'].add(exception_type)
                    results['exception_details'][exception_type].append(detail)
                    results['stats']['exception'] += 1


def init_results(file_path: str) -> Dict[str, Any]:
    """初始化结果字典"""
    return {
        'filename': os.path.basename(file_path),
        'stats': defaultdict(int),
        'keyword_counts': defaultdict(int),
        'exception_counts': defaultdict(int),
        'error_times': [],
        'error_details': [],
        'exception_details': defaultdict(list),
        'time_plot': '',
        'all_keywords': set(),  # 新增：记录文件中出现的所有关键词
        'all_exceptions': set()
    }


def process_line(line: str, line_number: int, results: Dict[str, Any], user_id, TIME_REGEX, KEYWORD_REGEXES,
                 EXCEPTION_REGEX, EXCEPTION_PATTERNS):
    """处理单行日志"""
    user_keywords = kwrods(user_id)  # 预先获取
    line = clean_text(line)
    results['stats']['lines'] += 1

    # 使用预编译的正则表达式检查关键词
    found_keywords = [kw for kw, regex in KEYWORD_REGEXES.items() if regex.search(line)]

    if found_keywords:
        results['stats']['keywords'] += len(found_keywords)

        for keyword in found_keywords:
            results['keyword_counts'][keyword] += 1
            results['all_keywords'].add(keyword)  # 记录出现的关键词

        # 'ERROR' 则 error_details
        # 使用预编译的正则提取时间戳
        match = TIME_REGEX.search(line)
        error_time = match.group(1) if match else '未知时间'

        # 如果是明确的错误信息这进去error_detail, 明确是异常，则进去exception_details
        for keyword in found_keywords:
            detail = {
                'source': results['filename'],
                'line_number': line_number,
                'time': error_time,
                'keyword': keyword,
                'message': line.strip(),
            }
            if keyword in user_keywords:
                results['error_details'].append(detail)

                results['stats']['errors'] += 1
                if match:
                    try:
                        error_time = datetime.strptime(match.group(1), '%Y-%m-%d%H:%M:%S.%f')
                        results['error_times'].append(error_time)
                    except ValueError:
                        pass
                    finally:
                        pass

                break
            else:
                # 异常类型识别
                exception_match = EXCEPTION_REGEX.search(line)
                # 常见异常匹配 1、匹配  2、不匹配

                exception_type = exception_match.group() if exception_match else EXCEPTION_PATTERNS['OtherError']
                detail['exception'] = exception_type
                results['exception_counts'][exception_type] += 1
                results['all_exceptions'].add(exception_type)
                results['exception_details'][exception_type].append(detail)
                results['stats']['exception'] += 1


def generate_file_plots(results: Dict[str, Any]):
    """生成单个文件的图表，优化性能"""
    if not results['error_times']:
        return

    # 使用采样减少数据点数量
    error_times = pd.DataFrame({'time': results['error_times']})
    if len(error_times) > 1000:
        error_times = error_times.sample(1000)

    error_times['hour'] = error_times['time'].dt.hour
    hourly_errors = error_times.groupby('hour').size().reset_index(name='count')

    fig = px.bar(hourly_errors, x='hour', y='count',
                 title='每小时错误分布',
                 labels={'hour': '小时', 'count': '错误数量'})
    results['time_plot'] = fig.to_html(full_html=False, include_plotlyjs='cdn')


def clean_text(text: str) -> str:
    """清理文本中的非法字符"""
    if not isinstance(text, str):
        return str(text)
    return text.encode('utf-8', errors='replace').decode('utf-8')


def merge_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并所有文件的分析结果"""
    merged = {
        'file_count': len(all_results),
        'file_list': [],
        'total_lines': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'total_keywords': 0,
        'keyword_counts': defaultdict(int),
        'error_times': [],
        'earliest_error': None,
        'latest_error': None
    }

    for result in all_results:
        merged['file_list'].append({
            'filename': result['filename'],
            'stats': dict(result['stats'])
        })
        merged['total_lines'] += result['stats']['lines']
        merged['total_errors'] += result['stats']['errors']
        merged['total_warnings'] += result['stats']['warnings']
        merged['total_keywords'] += result['stats']['keywords']

        for k, v in result['keyword_counts'].items():
            merged['keyword_counts'][k] += v

        merged['error_times'].extend(result['error_times'])

    # 计算最早和最晚错误时间
    if merged['error_times']:
        merged['earliest_error'] = min(merged['error_times']).strftime('%Y-%m-%d%H:%M:%S.%f')
        merged['latest_error'] = max(merged['error_times']).strftime('%Y-%m-%d%H:%M:%S.%f')

    return merged


def generate_plots(merged_results: Dict[str, Any]):
    """生成综合可视化图表"""
    plots = {}

    # 综合错误时间分布图
    if merged_results['error_times']:
        error_times = pd.DataFrame({'time': merged_results['error_times']})
        error_times['hour'] = error_times['time'].dt.hour
        hourly_errors = error_times.groupby('hour').size().reset_index(name='count')

        time_fig = px.bar(hourly_errors, x='hour', y='count',
                          title='综合每小时错误数量分布',
                          labels={'hour': '小时', 'count': '错误数量'})
        plots['time_plot'] = time_fig.to_html(full_html=False)
    else:
        plots['time_plot'] = "<p>未发现错误日志</p>"

    # 关键词分布图
    if merged_results['keyword_counts']:
        keywords_df = pd.DataFrame({
            'keyword': list(merged_results['keyword_counts'].keys()),
            'count': list(merged_results['keyword_counts'].values())
        }).sort_values('count', ascending=False)

        keyword_fig = px.bar(keywords_df, x='keyword', y='count',
                             title='关键词出现频率',
                             labels={'keyword': '关键词', 'count': '出现次数'})
        plots['keyword_plot'] = keyword_fig.to_html(full_html=False)
    else:
        plots['keyword_plot'] = "<p>未发现关键词匹配</p>"

    return plots


def generate_html_report_task(app_context, REPORT_FOLDER, file_results: List[Dict[str, Any]],
                         merged_results: Dict[str, Any], user_id, batch_id):
    """生成HTML报告"""
    if not current_app:
        app = create_app()
        ctx = app.app_context()
        ctx.push()
    else:
        ctx = None

    try:
        with app_context:
            with progress_lock:  # Ensure thread-safe operation
                generate_html_report(REPORT_FOLDER, file_results, merged_results, user_id, batch_id)
    except Exception as e:
        logger.error("生成报告失败{}".format(e))
    finally:
        if ctx:
            ctx.pop()



def generate_html_report(REPORT_FOLDER, file_results: List[Dict[str, Any]],
                         merged_results: Dict[str, Any], user_id, batch_id):
    try:
        # 准备数据
        KEYWORDS = kwrods(user_id)
        total_keywords = sum(merged_results['keyword_counts'].values()) or 1
        all_exceptions = set()
        all_keywords = set()
        for file in file_results:
            all_keywords.update(file['all_keywords'])
            all_exceptions.update(file['all_exceptions'])

        sorted_keywords = sorted(all_keywords)

        # 准备全局搜索数据
        global_errors = []
        for file in file_results:
            global_errors.extend(file['error_details'])
            global_errors.extend(file['exception_details'])

        # 创建自定义过滤器
        def replace_keywords(text):
            return highlight_keywords(text, KEYWORDS)

        filters.FILTERS['replace_keywords'] = replace_keywords
        # 生成综合图表
        plots = generate_plots(merged_results)

        # 准备异常统计数据
        exception_counts = defaultdict(int)
        exception_details = defaultdict(list)
        exception_first_occurrence = {}
        exception_last_occurrence = {}

        # 合并所有关键词计数
        for file in file_results:
            for exc_type, count in file['exception_counts'].items():
                exception_counts[exc_type] += count
            for exc_type, details in file['exception_details'].items():
                exception_details[exc_type].extend(details)

                # 记录首次和最后出现时间
                if details:
                    times = [d['time'] for d in details if d['time'] != '未知时间']
                    if times:
                        if exc_type not in exception_first_occurrence or times[0] < exception_first_occurrence[exc_type]:
                            exception_first_occurrence[exc_type] = min(times)
                        if exc_type not in exception_last_occurrence or times[-1] > exception_last_occurrence[exc_type]:
                            exception_last_occurrence[exc_type] = max(times)

        context = {
            'report_time': datetime.now().strftime('%Y-%m-%d%H:%M:%S.%f'),
            'merged_results': {
                'file_count': len('file_results'),
                'total_lines': sum(f['stats']['lines'] for f in file_results),
                'total_errors': sum(f['stats']['errors'] for f in file_results),
                'total_warnings': sum(f['stats']['warnings'] for f in file_results),
                'total_keywords': sum(f['stats']['keywords'] for f in file_results),
                'keyword_counts': defaultdict(int,
                                              {k: sum(f['keyword_counts'][k] for f in file_results) for k in KEYWORDS}),
                'exception_counts': dict(sorted(exception_counts.items(), key=lambda x: x[1], reverse=True)),
                'exception_details': exception_details,
                'exception_first_occurrence': exception_first_occurrence,
                'exception_last_occurrence': exception_last_occurrence
            },
            'file_list': merged_results['file_list'],
            'file_results': file_results,
            'earliest_error': merged_results['earliest_error'],
            'latest_error': merged_results['latest_error'],
            'time_plot': plots.get('time_plot', ''),
            'keyword_plot': plots.get('keyword_plot', ''),
            'max_details': getConfig(user_id, 'performance', 'max_error_details'),
            'replace_keywords': replace_keywords,
            'MAX_ERROR_DETAILS': getConfig(user_id, 'performance', 'max_error_details'),
            'all_keywords': sorted_keywords,
            'global_errors': global_errors,
        }

        for file in file_results:
            for kw, count in file['keyword_counts'].items():
                context['merged_results']['keyword_counts'][kw] += count

        # 使用Flask的render_template渲染模板
        html_report = render_template('report_template.html', **context)

        # 保存报告

        name = f"log_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        report_file = os.path.join(REPORT_FOLDER, name)
        with open(report_file, 'w', encoding='utf-8', errors='xmlcharrefreplace') as f:
            f.write(html_report)

        # batch_id 存储 报告名称
        # update_report_path(batch_id, name, user_id)
        return name
    except Exception as e:
        logger.error("生成报告失败{}".format(e))


def analyze_file(file_path: str, analyzer: Callable[[str], Dict[str, Any]], user_id, batch_id) -> Dict[str, Any]:
    """包装函数用于多进程处理"""
    return analyzer(file_path, user_id, batch_id)


def remedialWorks(REPORT_FOLDER, EXCEPTION_PATTERNS, user_id, batch_id, file_results):
    try:

        def create_background_task(save_func, task_name):
            @copy_current_request_context
            def wrapper(*args, **kwargs):
                try:
                    save_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"存储日志文件失败 {task_name}: {e}")

            return wrapper

        background_save_file_stats_task = create_background_task(
            lambda bid, fr: save_file_stats_to_db(bid, fr, REPORT_FOLDER),
            "save_file_stats_to_db"
        )

        background_save_errors_task = create_background_task(
            lambda bid, fr: save_errors_to_db(bid, fr,
                                              getConfig(user_id, 'performance', 'batch_size')),
            "save_errors_to_db"
        )

        background_save_exceptions_task = create_background_task(
            lambda bid, fr: save_exceptions_to_db(bid, fr, EXCEPTION_PATTERNS,
                                                  getConfig(user_id, 'performance', 'batch_size')),
            "save_file_stats_to_db"
        )

        from concurrent.futures import ThreadPoolExecutor

        def process_in_background(batch_id, file_results, REPORT_FOLDER):
            tasks = [
                (background_save_file_stats_task, (batch_id, file_results, REPORT_FOLDER), {}),
                (background_save_exceptions_task, (batch_id, file_results), {}),
                (background_save_errors_task, (batch_id, file_results), {})
            ]
            # 使用线程池执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for func, args, kwargs in tasks:
                    try:
                        # 提交任务到线程池
                        future = executor.submit(func, *args, **kwargs)
                        futures.append(future)
                    except Exception as e:
                        logger.error(f"提交后台任务失败: {e}")

        process_in_background(batch_id, file_results, REPORT_FOLDER)

    except Exception as e:
        logger.info("存储每个文件信息最终结果", e)


# 优化点2：使用更高效的任务分配方式
def analyze_wrapper(args):
    analyzer_type, file, user_id, batch_id = args
    if analyzer_type == 'mmap':
        return analyze_with_mmap(file, user_id, batch_id)
    else:
        return analyze_with_buffer(file, user_id, batch_id)


def process(log_files, REPORT_FOLDER, user_id, batch_id):
    logger.info("开始日志分析...")
    report_path = None  # 初始化变量用于finally块
    start_time = datetime.now()
    try:
        # 0. 预加载配置（移到函数开始处）
        yappi_started = False
        if current_app.config.get('ENABLE_PROFILING', False):
            yappi.start()
            yappi_started = True

        # 1. 初始化批次信息（提前获取所有必要配置）
        with create_analysis_session(log_files, user_id, batch_id) as batch:
            logger.info(f"分析批次ID: {batch.batch_id}")
            logger.info(f"找到 {len(log_files)} 个日志文件，开始分析...")

            batch.status = 'parsing'
            db.session.commit()
            # 提前获取所有配置（减少重复调用）

            def get_safe_workers():
                try:
                    workers = int(getConfig(user_id, 'performance', 'max_workers'))
                    return max(1, min(workers, os.cpu_count() or 4))
                except (ValueError, KeyError):
                    return max(1, os.cpu_count() or 4)

            config = {
                'max_workers': min(
                    get_safe_workers(),
                    len(log_files)
                ),
                'min_mmap_size': getConfig(user_id, 'performance', 'min_mmap_size'),
                'exception_patterns': UserConfig.get_user_config(user_id, 'exception', 'patterns'),
            }

            # 2. 并行化文件预处理
            def prepare_file(file):
                return (
                    'mmap' if os.path.getsize(file) > config['min_mmap_size'] else 'buffer',
                    file,
                    user_id,
                    batch_id
                )

            # 使用线程池预处理文件信息
            with ThreadPoolExecutor(max_workers=config['max_workers']) as prep_pool:
                analyzers = list(prep_pool.map(prepare_file, log_files))

            # 3. 优化分析过程
            file_results = []
            with Pool(config['max_workers']) as analysis_pool:
                # 使用chunksize减少IPC开销
                for i, result in enumerate(
                        analysis_pool.imap_unordered(
                            analyze_wrapper,
                            analyzers,
                            chunksize=max(1, len(log_files) // (config['max_workers'] * 2))
                        )):
                    file_results.append(result)

                    # 进度更新优化（减少数据库提交次数）
                    if i % 20 == 0 or i == len(log_files) - 1:
                        progress = int((i + 1) * 100 / len(log_files))
                        batch.progress = progress
                        logger.info(f"进度: {progress}% ({i + 1}/{len(log_files)})")

                # 批量提交进度更新
                db.session.commit()
            try:
                # 优化点5：分批合并结果减少内存峰值
                merged_results = merge_results(file_results)
            except Exception as e:
                logger.info("合并结果异常", e)

            try:
                # 更新批次信息
                batch.total_lines = merged_results['total_lines']
                batch.total_errors = merged_results['total_errors']
                batch.total_warnings = merged_results['total_warnings']
            except Exception as e:
                logger.error(f"更新批次信息失败: {str(e)}", exc_info=True)

            try:
                # 生成报告
                report_path = generate_html_report(REPORT_FOLDER, file_results, merged_results, user_id, batch_id)
                logger.info(report_path)
                batch.report_path = report_path
                # 使用partial预先绑定参数
                # task = partial(
                #     generate_html_report_task,
                #     current_app.app_context(),
                #     REPORT_FOLDER,
                #     file_results,
                #     merged_results,
                #     user_id,
                #     batch_id
                # )
                #
                # # 使用with管理线程池
                # with ThreadPoolExecutor(max_workers=1) as executor:
                #     executor.submit(task)
            except Exception as e:
                batch.status = 'failed'
                db.session.commit()
                logger.error(f"生成报告失败: {str(e)}", exc_info=True)
                raise
            if yappi_started:
                yappi.stop()
            return {batch_id, report_path}
    except Exception as e:
        logger.error(f"日志分析过程失败: {str(e)}", exc_info=True)
        if 'batch' in locals():
            batch.status = 'failed'
            db.session.commit()
        raise
    finally:
        # 确保资源清理
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"分析完成! 总耗时: {elapsed:.2f} 秒")
        logger.info(f"报告已生成: {report_path}")

        # 优化点6：后台执行补救工作
        if 'file_results' in locals():
            try:
                # remedialWorks(REPORT_FOLDER, config['exception_patterns'], user_id, batch_id, file_results)
                # 使用线程池异步执行
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(
                        remedialWorks,
                        REPORT_FOLDER,
                        config['exception_patterns'],
                        user_id,
                        batch_id,
                        file_results
                    )
            except Exception as e:
                logger.error(f"补救工作执行失败: {str(e)}", exc_info=True)
