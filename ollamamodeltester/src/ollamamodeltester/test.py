"""
Ollama 本地大语言模型测试工具
使用 PySide6 创建的图形化界面，支持多模型批量测试和结果可视化
"""
import importlib.metadata
import sys
import json
import time
import csv
import threading
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QTableWidget, 
    QTableWidgetItem, QProgressBar, QTabWidget, QCheckBox,
    QSpinBox, QGroupBox, QSplitter, QFileDialog, QMessageBox,
    QScrollArea, QListWidget, QApplication
)

import ollama
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
import os
import subprocess


class TestScenario:
    """测试场景类"""
    def __init__(self, name: str, description: str, prompt_template: str, evaluation_criteria: str):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.evaluation_criteria = evaluation_criteria


class ModelTestWorker(QThread):
    """模型测试工作线程"""
    progress_updated = Signal(int)
    test_completed = Signal(dict)
    log_updated = Signal(str)
    status_updated = Signal(str)
    single_test_completed = Signal(dict)
    stream_updated = Signal(str)
    
    def __init__(self, models: List[str], test_pairs: List[tuple], ollama_client):
        super().__init__()
        self.models = models
        self.test_pairs = test_pairs
        self.results = []
        self.ollama_client = ollama_client
        self.is_running = True
        
    def run(self):
        """执行测试主循环"""
        total_tests = len(self.models) * len(self.test_pairs)
        completed = 0
        
        self.log_updated.emit(f"开始测试 {len(self.models)} 个模型，{len(self.test_pairs)} 个场景")
        
        for model in self.models:
            if not self.is_running:
                break
                
            for scenario, test_input in self.test_pairs:
                if not self.is_running:
                    break
                    
                try:
                    status_msg = f"正在测试: {model} | 场景: {scenario.name} | 进度: {completed+1}/{total_tests}"
                    self.status_updated.emit(status_msg)
                    self.log_updated.emit(f"测试模型: {model}, 场景: {scenario.name}")
                    
                    result = self.run_single_test(model, scenario, test_input)
                    self.results.append(result)
                    
                    # 发送单个测试结果
                    self.single_test_completed.emit(result)
                    
                    completed += 1
                    progress = int((completed / total_tests) * 100)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    self.log_updated.emit(f"测试出错: {str(e)}")
                    
        self.test_completed.emit({'results': self.results})
    
    def stop(self):
        """停止测试"""
        self.is_running = False
    
    def run_single_test(self, model: str, scenario: TestScenario, test_input: str) -> Dict[str, Any]:
        """运行单个测试"""
        prompt = scenario.prompt_template.format(input=test_input)
        start_time = time.time()
        
        try:
            # 启用流式输出
            if self.ollama_client:
                stream = self.ollama_client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True
                )
            else:
                stream = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True
                )
            
            output = ""
            # 逐块读取流
            for chunk in stream:
                try:
                    if isinstance(chunk, dict):
                        msg = chunk.get('message') or {}
                        piece = msg.get('content', '')
                        if piece:
                            output += piece
                            # 实时推送到UI
                            self.stream_updated.emit(f"[{model} | {scenario.name}] {piece}")
                    else:
                        # 不同实现下可能直接返回字符串
                        s = str(chunk)
                        if s:
                            output += s
                            self.stream_updated.emit(f"[{model} | {scenario.name}] {s}")
                except Exception as ie:
                    # 单个分片解析失败不应中断整个生成
                    self.log_updated.emit(f"分片解析异常: {ie}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 计算性能指标
            estimated_tokens = len(output.split())
            tokens_per_second = estimated_tokens / duration if duration > 0 else 0
            
            # 模型自评分
            self_score = self.get_self_evaluation(model, scenario, test_input, output)
            
            return {
                'model': model,
                'scenario': scenario.name,
                'input': test_input,
                'output': output,
                'duration': duration,
                'tokens_per_second': tokens_per_second,
                'self_score': self_score,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'model': model,
                'scenario': scenario.name,
                'input': test_input,
                'output': f'错误: {str(e)}',
                'duration': 0,
                'tokens_per_second': 0,
                'self_score': 0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    def get_self_evaluation(self, model: str, scenario: TestScenario, input_text: str, output: str) -> float:
        """获取模型自评分"""
        eval_prompt = f"""
        请对以下回答进行评分(1-10分):
        
        任务: {scenario.description}
        输入: {input_text}
        输出: {output}
        
        评分标准: {scenario.evaluation_criteria}
        
        请只返回一个1-10之间的数字分数，不要其他解释。
        """
        
        try:
            if self.ollama_client:
                response = self.ollama_client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': eval_prompt}]
                )
            else:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': eval_prompt}]
                )
            
            score_text = response['message']['content'].strip()
            # 提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?', score_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 1), 10)  # 确保在1-10范围内
            return 5.0
        except:
            return 5.0


class MatplotlibWidget(QWidget):
    """Matplotlib图表组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_performance_comparison(self, results_df: pd.DataFrame):
        """绘制性能对比图"""
        if results_df.empty:
            return
            
        self.figure.clear()
        
        # 创建子图 - 2x2 布局
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        
        try:
            # 1. Tokens/s 对比
            model_performance = results_df.groupby('model')['tokens_per_second'].mean()
            ax1.bar(model_performance.index, model_performance.values, color='skyblue')
            ax1.set_title('平均 Tokens/s', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 2. 响应时间对比
            model_duration = results_df.groupby('model')['duration'].mean()
            ax2.bar(model_duration.index, model_duration.values, color='lightcoral')
            ax2.set_title('平均响应时间 (秒)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. 自评分对比
            model_scores = results_df.groupby('model')['self_score'].mean()
            ax3.bar(model_scores.index, model_scores.values, color='lightgreen')
            ax3.set_title('平均自评分', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_ylim(0, 10)
            ax3.grid(True, alpha=0.3)
            
            # 4. 场景表现对比
            scenario_scores = results_df.groupby('scenario')['self_score'].mean()
            ax4.bar(scenario_scores.index, scenario_scores.values, color='gold')
            ax4.set_title('各场景平均得分', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 10)
            ax4.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            # 如果绘图失败，显示错误信息
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'绘图错误: {str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.canvas.draw()


class OllamaModelTester(QMainWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.test_worker = None
        self.results_data = []
        self.ollama_client = None
        self.checked_model_order = []
        
        self.init_test_scenarios()
        self.init_ui()
        self.init_ollama_connection()
        self.load_available_models()
    
    def init_test_scenarios(self):
        """初始化测试场景"""
        self.test_scenarios = [
            TestScenario(
                "自然语言转代码",
                "将自然语言描述转换为问题中所说的代码",
                "请将以下自然语言描述转换为问题中所说的代码，只返回代码，不要解释:\n{input}",
                "代码正确性、可读性、效率"
            ),
            TestScenario(
                "中英互译",
                "中英文双向翻译",
                "请翻译以下文本，保持原意，语言自然流畅:\n{input}",
                "翻译准确性、流畅性、语言地道性"
            ),
            TestScenario(
                "代码解释",
                "解释代码功能和逻辑",
                "请详细解释以下代码的功能和逻辑:\n{input}",
                "解释准确性、清晰度、完整性"
            ),
            TestScenario(
                "问题解答",
                "回答技术问题",
                "请详细回答以下问题:\n{input}",
                "答案准确性、详细程度、实用性"
            )
        ]
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Ollama 本地大语言模型测试工具")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧结果面板
        results_panel = self.create_results_panel()
        splitter.addWidget(results_panel)
        
        # 设置分割器比例
        splitter.setSizes([450, 1150])
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 创建菜单栏
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        export_action = QtGui.QAction('导出结果到CSV', self)
        export_action.setShortcut('Ctrl+S')
        export_action.triggered.connect(self.export_to_csv)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QtGui.QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        refresh_action = QtGui.QAction('刷新模型列表', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.load_available_models)
        tools_menu.addAction(refresh_action)
        connect_action = QtGui.QAction('重新连接Ollama', self)
        connect_action.triggered.connect(self.init_ollama_connection)
        tools_menu.addAction(connect_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QtGui.QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 模型列表
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.model_list.itemChanged.connect(self.on_model_item_changed)
        model_layout.addWidget(self.model_list)
        
        # 模型操作按钮
        model_buttons = QHBoxLayout()
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.load_available_models)
        model_buttons.addWidget(refresh_btn)
        
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_models)
        model_buttons.addWidget(select_all_btn)
        
        clear_all_btn = QPushButton("清空")
        clear_all_btn.clicked.connect(self.clear_all_models)
        model_buttons.addWidget(clear_all_btn)
        
        model_layout.addLayout(model_buttons)
        layout.addWidget(model_group)
        
        # 测试场景组
        scenario_group = QGroupBox("测试场景")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_checkboxes = []
        for scenario in self.test_scenarios:
            checkbox = QCheckBox(scenario.name)
            checkbox.setChecked(True)
            checkbox.setToolTip(scenario.description)
            self.scenario_checkboxes.append(checkbox)
            scenario_layout.addWidget(checkbox)
        
        layout.addWidget(scenario_group)
        
        # 测试输入组
        input_group = QGroupBox("测试输入")
        input_layout = QVBoxLayout(input_group)
        
        # 滚动区域用于容纳所有输入框
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.scenario_inputs = {}
        default_inputs = {
            "自然语言转代码": "使用python计算斐波那契数列的第10项",
            "中英互译": "Hello, how are you today? I hope you're having a great day!",
            "代码解释": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "问题解答": "什么是机器学习中的过拟合现象？如何避免？"
        }
        
        for scenario in self.test_scenarios:
            scenario_label = QLabel(f"{scenario.name}:")
            scenario_label.setFont(QtGui.QFont("", 9, QtGui.QFont.Bold))
            scroll_layout.addWidget(scenario_label)
            
            scenario_input = QTextEdit()
            scenario_input.setMaximumHeight(80)
            scenario_input.setPlainText(default_inputs.get(scenario.name, ""))
            scenario_input.setToolTip(f"为 {scenario.name} 场景输入测试内容")
            
            self.scenario_inputs[scenario.name] = scenario_input
            scroll_layout.addWidget(scenario_input)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        input_layout.addWidget(scroll_area)
        layout.addWidget(input_group)
        
        # 控制按钮组
        button_group = QGroupBox("测试控制")
        button_layout = QVBoxLayout(button_group)
        
        self.start_btn = QPushButton("开始批量测试")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.start_btn.clicked.connect(self.start_testing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止测试")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        self.stop_btn.clicked.connect(self.stop_testing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        button_layout.addWidget(self.progress_bar)
        
        layout.addWidget(button_group)
        
        return panel
    
    def create_results_panel(self):
        """创建结果面板"""
        panel = QTabWidget()
        
        # 结果表格标签页
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        # 表格工具栏
        table_toolbar = QHBoxLayout()
        
        export_btn = QPushButton("导出CSV")
        export_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        export_btn.clicked.connect(self.export_to_csv)
        table_toolbar.addWidget(export_btn)
        
        clear_btn = QPushButton("清空结果")
        clear_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogDiscardButton))
        clear_btn.clicked.connect(self.clear_results)
        table_toolbar.addWidget(clear_btn)
        
        table_toolbar.addStretch()
        
        # 结果统计标签
        self.stats_label = QLabel("总计: 0 个结果")
        table_toolbar.addWidget(self.stats_label)
        
        table_layout.addLayout(table_toolbar)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.setup_results_table()
        table_layout.addWidget(self.results_table)
        
        panel.addTab(table_widget, "测试结果")
        
        # 可视化标签页
        self.chart_widget = MatplotlibWidget()
        panel.addTab(self.chart_widget, "性能可视化")
        
        # 日志标签页
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        log_toolbar = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_toolbar.addWidget(clear_log_btn)
        log_toolbar.addStretch()
        log_layout.addLayout(log_toolbar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        panel.addTab(log_widget, "测试日志")
        
        return panel
    
    def setup_results_table(self):
        """设置结果表格"""
        headers = ["模型", "场景", "输入", "输出", "耗时(秒)", "Tokens/s", "自评分", "时间戳", "状态"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # 调整列宽
        self.results_table.setColumnWidth(0, 130)  # 模型
        self.results_table.setColumnWidth(1, 130)  # 场景
        self.results_table.setColumnWidth(2, 200)  # 输入
        self.results_table.setColumnWidth(3, 400)  # 输出
        self.results_table.setColumnWidth(4, 80)   # 耗时
        self.results_table.setColumnWidth(5, 80)   # Tokens/s
        self.results_table.setColumnWidth(6, 80)   # 自评分
        self.results_table.setColumnWidth(7, 160)  # 时间戳
        self.results_table.setColumnWidth(8, 80)   # 状态
        
        # 设置表格属性
        self.results_table.setWordWrap(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    
    def init_ollama_connection(self):
        """初始化Ollama连接"""
        try:
            self.ollama_host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
            
            # 测试连接
            resp = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if resp.status_code == 200:
                self.ollama_client = ollama.Client(host=self.ollama_host)
                self.ollama_client.list()  # 验证客户端可用
                self.log_message(f"Ollama连接成功: {self.ollama_host}")
                self.statusBar().showMessage(f"已连接到 Ollama: {self.ollama_host}")
            else:
                raise RuntimeError(f"服务响应异常，状态码: {resp.status_code}")
                
        except Exception as e:
            self.log_message(f"Ollama连接失败: {str(e)}")
            self.statusBar().showMessage("Ollama连接失败")
            self.ollama_client = None
            QMessageBox.warning(self, "连接失败", f"无法连接到Ollama服务:\n{str(e)}")
    
    def load_available_models(self):
        """加载可用的Ollama模型（优先HTTP/Client，失败回退CLI），并以复选框形式呈现"""
        self.model_list.clear()
        self.checked_model_order.clear()
        # 先尝试通过 Client/HTTP 获取
        try:
            if not self.check_ollama_connection():
                raise RuntimeError("Ollama服务不可用")
            models = self.ollama_client.list() if self.ollama_client else ollama.list()
            model_objs = []
            if isinstance(models, dict) and 'models' in models:
                model_objs = models['models']
            elif isinstance(models, list):
                model_objs = models
            else:
                raise RuntimeError(f"未知的模型列表响应格式: {models}")
            names = []
            for m in model_objs:
                name = None
                if isinstance(m, dict):
                    for key in ('name', 'model', 'id', 'title'):
                        if key in m and m[key]:
                            name = m[key]
                            break
                elif isinstance(m, str):
                    name = m
                else:
                    name = str(m)
                if name:
                    names.append(name)
            names = sorted(set(names))
            # 打印并填充
            print("=== 模型列表（HTTP/Client） ===")
            # 成功通过HTTP/Client获取到 names 之后，改为创建复选框项
            for n in names:
                print(n)
                item = QtWidgets.QListWidgetItem(n)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.model_list.addItem(item)
            self.log_message(f"已加载 {len(names)} 个模型（HTTP/Client）")
            return
        except Exception as e_http:
            self.log_message(f"通过HTTP/Client获取模型失败: {str(e_http)}，尝试CLI回退")
            try:
                cmd = "ollama list | sort"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10
                )
                output = result.stdout.strip()
                if result.returncode != 0 or not output:
                    result = subprocess.run(
                        ["ollama", "list"], shell=False, capture_output=True, text=True, timeout=10
                    )
                    output = result.stdout.strip()
                print("=== 模型列表（CLI 原始输出） ===")
                print(output)
                self.log_message("已打印CLI原始输出，请查看控制台。")
                # 解析表格输出
                names = []
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    upper = line.upper()
                    if "NAME" in upper and ("MODIFIED" in upper or "SIZE" in upper):
                        continue
                    first = line.split()[0]
                    names.append(first)
                names = sorted(set(names))
                # CLI 回退路径中，成功解析 names 后同样改为复选框项
                self.model_list.clear()
                for n in names:
                    item = QtWidgets.QListWidgetItem(n)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    self.model_list.addItem(item)
                self.log_message(f"已加载 {len(names)} 个模型（CLI）")
            except FileNotFoundError:
                self.log_message("未找到 ollama 可执行文件，请确认已安装并加入 PATH。")
                QMessageBox.warning(self, "错误", "未找到 ollama 可执行文件，请确认已安装并加入 PATH。")
            except Exception as e_cli:
                self.log_message(f"通过CLI获取模型失败: {str(e_cli)}")
                QMessageBox.warning(self, "错误", f"无法获取模型列表: {str(e_cli)}")
    
    def select_all_models(self):
        """选择所有模型"""
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setCheckState(QtCore.Qt.Checked)
    
    def clear_all_models(self):
        """清空所有模型选择"""
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setCheckState(QtCore.Qt.Unchecked)
    
    def on_model_item_changed(self, item):
        """处理模型选择变化"""
        model_name = item.text()
        
        if item.checkState() == QtCore.Qt.Checked:
            if model_name not in self.checked_model_order:
                self.checked_model_order.append(model_name)
                self.log_message(f"选中模型: {model_name}")
        else:
            if model_name in self.checked_model_order:
                self.checked_model_order.remove(model_name)
                self.log_message(f"取消选中模型: {model_name}")
        
        selected_count = len(self.checked_model_order)
        self.statusBar().showMessage(f"已选择 {selected_count} 个模型")
    
    def start_testing(self):
        """开始测试"""
        # 获取选中的模型
        selected_models = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selected_models.append(item.text())
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return
        
        # 获取选中的场景和对应输入
        test_pairs = []
        for i, scenario in enumerate(self.test_scenarios):
            if self.scenario_checkboxes[i].isChecked():
                input_text = self.scenario_inputs[scenario.name].toPlainText().strip()
                if input_text:
                    test_pairs.append((scenario, input_text))
        
        if not test_pairs:
            QMessageBox.warning(self, "警告", "请至少选择一个测试场景并提供输入")
            return
        
        if not self.ollama_client:
            QMessageBox.critical(self, "错误", "Ollama连接未建立，请检查连接")
            return
        
        # 按用户勾选顺序排序模型
        ordered_models = []
        for model in self.checked_model_order:
            if model in selected_models:
                ordered_models.append(model)
        for model in selected_models:
            if model not in ordered_models:
                ordered_models.append(model)
        
        # 创建并启动测试线程
        self.test_worker = ModelTestWorker(ordered_models, test_pairs, self.ollama_client)
        self.test_worker.progress_updated.connect(self.progress_bar.setValue)
        self.test_worker.test_completed.connect(self.on_test_completed)
        self.test_worker.log_updated.connect(self.log_message)
        self.test_worker.status_updated.connect(self.statusBar().showMessage)
        self.test_worker.single_test_completed.connect(self.on_single_test_completed)
        self.test_worker.stream_updated.connect(self.on_stream_updated)
        
        # 更新UI状态
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_message("=" * 50)
        self.log_message(f"开始测试 - 模型: {len(ordered_models)}个, 场景: {len(test_pairs)}个")
        self.log_message("=" * 50)
        
        self.test_worker.start()
    
    def stop_testing(self):
        """停止测试"""
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.stop()
            self.test_worker.wait(3000)  # 等待3秒
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("测试已停止")
        self.log_message("测试已停止")
    
    def on_single_test_completed(self, result: Dict[str, Any]):
        """单个测试完成回调"""
        self.results_data.append(result)
        
        # 添加到表格
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # 创建表格项
        items = [
            QTableWidgetItem(result['model']),
            QTableWidgetItem(result['scenario']),
            QTableWidgetItem(result['input'][:100] + "..." if len(result['input']) > 100 else result['input']),
            QTableWidgetItem(result['output'][:200] + "..." if len(result['output']) > 200 else result['output']),
            QTableWidgetItem(f"{result['duration']:.2f}"),
            QTableWidgetItem(f"{result['tokens_per_second']:.2f}"),
            QTableWidgetItem(f"{result['self_score']:.1f}"),
            QTableWidgetItem(result['timestamp'].split('T')[1][:8]),  # 只显示时间
            QTableWidgetItem(result['status'])
        ]
        
        # 设置状态颜色
        if result['status'] == 'success':
            items[-1].setBackground(QtGui.QColor(144, 238, 144))  # 浅绿色
        else:
            items[-1].setBackground(QtGui.QColor(255, 182, 193))  # 浅红色
        
        # 添加到表格
        for col, item in enumerate(items):
            self.results_table.setItem(row, col, item)
        
        # 调整行高并滚动到底部
        self.results_table.resizeRowToContents(row)
        self.results_table.scrollToBottom()
        
        # 更新统计
        self.update_stats()
        
        # 实时更新可视化
        self.update_visualization()
    
    def on_test_completed(self, data: Dict[str, Any]):
        """测试完成回调"""
        results = data['results']
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        self.statusBar().showMessage(f"测试完成! 成功: {success_count}/{len(results)}")
        
        self.log_message("=" * 50)
        self.log_message(f"测试完成! 总计: {len(results)}个, 成功: {success_count}个")
        self.log_message("=" * 50)
        
        self.update_visualization()
    
    def update_stats(self):
        """更新统计信息"""
        total = len(self.results_data)
        success = sum(1 for r in self.results_data if r['status'] == 'success')
        self.stats_label.setText(f"总计: {total} 个结果 (成功: {success}, 失败: {total-success})")
    
    def update_visualization(self):
        """更新可视化图表"""
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            # 只显示成功的结果
            success_df = df[df['status'] == 'success']
            if not success_df.empty:
                self.chart_widget.plot_performance_comparison(success_df)
    
    def export_to_csv(self):
        """导出结果到CSV"""
        if not self.results_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ollama_test_results_{timestamp}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", default_filename, "CSV文件 (*.csv)"
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.results_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"结果已导出到:\n{filename}")
                self.log_message(f"结果已导出到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def check_ollama_connection(self):
        """检查Ollama连接状态"""
        try:
            if self.ollama_client:
                self.ollama_client.list()
                return True
            return False
        except Exception:
            return False
    
    def clear_results(self):
        """清空结果"""
        if not self.results_data:
            return
            
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有结果吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.results_data.clear()
            self.results_table.setRowCount(0)
            self.chart_widget.figure.clear()
            self.chart_widget.canvas.draw()
            self.update_stats()
            self.log_message("结果已清空")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_stream_updated(self, text: str):
        """将流式分片直接追加到日志输出（不加时间戳，保证连续性）"""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertPlainText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
            "Ollama 本地大语言模型测试工具\n\n"
            "功能特点:\n"
            "• 多模型批量测试\n"
            "• 自然语言转代码\n"
            "• 中英文互译\n"
            "• 代码解释\n"
            "• 技术问答\n"
            "• 性能可视化\n"
            "• CSV数据导出\n\n"
            "支持 Briefcase 和直接 Python 运行")
    
    def closeEvent(self, event):
        """应用关闭事件"""
        if self.test_worker and self.test_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出", "测试正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.test_worker.stop()
                self.test_worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有结果吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.results_data.clear()
            self.results_table.setRowCount(0)
            self.chart_widget.figure.clear()
            self.chart_widget.canvas.draw()
            self.update_stats()
            self.log_message("结果已清空")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_stream_updated(self, text: str):
        """将流式分片直接追加到日志输出（不加时间戳，保证连续性）"""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertPlainText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
            "Ollama 本地大语言模型测试工具\n\n"
            "功能特点:\n"
            "• 多模型批量测试\n"
            "• 自然语言转代码\n"
            "• 中英文互译\n"
            "• 代码解释\n"
            "• 技术问答\n"
            "• 性能可视化\n"
            "• CSV数据导出\n\n"
            "支持 Briefcase 和直接 Python 运行")
    
    def closeEvent(self, event):
        """应用关闭事件"""
        if self.test_worker and self.test_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出", "测试正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.test_worker.stop()
                self.test_worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数 - 支持 Briefcase 和直接运行"""
    # 设置应用程序名称
    app_module = sys.modules["__main__"].__package__
    
    try:
        if app_module:
            # Briefcase 模式
            metadata = importlib.metadata.metadata(app_module)
            app_name = metadata["Formal-Name"]
        else:
            # 直接运行模式
            app_name = "OllamaModelTester"
    except (ValueError, importlib.metadata.PackageNotFoundError):
        app_name = "OllamaModelTester"
    
    QApplication.setApplicationName(app_name)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式
    
    # 设置应用图标（如果有的话）
    try:
        app.setWindowIcon(QtGui.QIcon("icon.ico"))
    except:
        pass
    
    # 创建主窗口
    main_window = OllamaModelTester()
    main_window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


class OllamaModelTester(QMainWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.test_worker = None
        self.results_data = []
        self.ollama_client = None
        self.checked_model_order = []
        
        self.init_test_scenarios()
        self.init_ui()
        self.init_ollama_connection()
        self.load_available_models()
    
    def init_test_scenarios(self):
        """初始化测试场景"""
        self.test_scenarios = [
            TestScenario(
                "自然语言转代码",
                "将自然语言描述转换为问题中所说的代码",
                "请将以下自然语言描述转换为问题中所说的代码，只返回代码，不要解释:\n{input}",
                "代码正确性、可读性、效率"
            ),
            TestScenario(
                "中英互译",
                "中英文双向翻译",
                "请翻译以下文本，保持原意，语言自然流畅:\n{input}",
                "翻译准确性、流畅性、语言地道性"
            ),
            TestScenario(
                "代码解释",
                "解释代码功能和逻辑",
                "请详细解释以下代码的功能和逻辑:\n{input}",
                "解释准确性、清晰度、完整性"
            ),
            TestScenario(
                "问题解答",
                "回答技术问题",
                "请详细回答以下问题:\n{input}",
                "答案准确性、详细程度、实用性"
            )
        ]
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Ollama 本地大语言模型测试工具")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧结果面板
        results_panel = self.create_results_panel()
        splitter.addWidget(results_panel)
        
        # 设置分割器比例
        splitter.setSizes([450, 1150])
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 创建菜单栏
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        export_action = QtGui.QAction('导出结果到CSV', self)
        export_action.setShortcut('Ctrl+S')
        export_action.triggered.connect(self.export_to_csv)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QtGui.QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        refresh_action = QtGui.QAction('刷新模型列表', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.load_available_models)
        tools_menu.addAction(refresh_action)
        connect_action = QtGui.QAction('重新连接Ollama', self)
        connect_action.triggered.connect(self.init_ollama_connection)
        tools_menu.addAction(connect_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QtGui.QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 模型列表
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.model_list.itemChanged.connect(self.on_model_item_changed)
        model_layout.addWidget(self.model_list)
        
        # 模型操作按钮
        model_buttons = QHBoxLayout()
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.load_available_models)
        model_buttons.addWidget(refresh_btn)
        
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_models)
        model_buttons.addWidget(select_all_btn)
        
        clear_all_btn = QPushButton("清空")
        clear_all_btn.clicked.connect(self.clear_all_models)
        model_buttons.addWidget(clear_all_btn)
        
        model_layout.addLayout(model_buttons)
        layout.addWidget(model_group)
        
        # 测试场景组
        scenario_group = QGroupBox("测试场景")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_checkboxes = []
        for scenario in self.test_scenarios:
            checkbox = QCheckBox(scenario.name)
            checkbox.setChecked(True)
            checkbox.setToolTip(scenario.description)
            self.scenario_checkboxes.append(checkbox)
            scenario_layout.addWidget(checkbox)
        
        layout.addWidget(scenario_group)
        
        # 测试输入组
        input_group = QGroupBox("测试输入")
        input_layout = QVBoxLayout(input_group)
        
        # 滚动区域用于容纳所有输入框
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.scenario_inputs = {}
        default_inputs = {
            "自然语言转代码": "使用python计算斐波那契数列的第10项",
            "中英互译": "Hello, how are you today? I hope you're having a great day!",
            "代码解释": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "问题解答": "什么是机器学习中的过拟合现象？如何避免？"
        }
        
        for scenario in self.test_scenarios:
            scenario_label = QLabel(f"{scenario.name}:")
            scenario_label.setFont(QtGui.QFont("", 9, QtGui.QFont.Bold))
            scroll_layout.addWidget(scenario_label)
            
            scenario_input = QTextEdit()
            scenario_input.setMaximumHeight(80)
            scenario_input.setPlainText(default_inputs.get(scenario.name, ""))
            scenario_input.setToolTip(f"为 {scenario.name} 场景输入测试内容")
            
            self.scenario_inputs[scenario.name] = scenario_input
            scroll_layout.addWidget(scenario_input)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        input_layout.addWidget(scroll_area)
        layout.addWidget(input_group)
        
        # 控制按钮组
        button_group = QGroupBox("测试控制")
        button_layout = QVBoxLayout(button_group)
        
        self.start_btn = QPushButton("开始批量测试")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.start_btn.clicked.connect(self.start_testing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止测试")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        self.stop_btn.clicked.connect(self.stop_testing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        button_layout.addWidget(self.progress_bar)
        
        layout.addWidget(button_group)
        
        return panel
    
    def create_results_panel(self):
        """创建结果面板"""
        panel = QTabWidget()
        
        # 结果表格标签页
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        # 表格工具栏
        table_toolbar = QHBoxLayout()
        
        export_btn = QPushButton("导出CSV")
        export_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        export_btn.clicked.connect(self.export_to_csv)
        table_toolbar.addWidget(export_btn)
        
        clear_btn = QPushButton("清空结果")
        clear_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogDiscardButton))
        clear_btn.clicked.connect(self.clear_results)
        table_toolbar.addWidget(clear_btn)
        
        table_toolbar.addStretch()
        
        # 结果统计标签
        self.stats_label = QLabel("总计: 0 个结果")
        table_toolbar.addWidget(self.stats_label)
        
        table_layout.addLayout(table_toolbar)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.setup_results_table()
        table_layout.addWidget(self.results_table)
        
        panel.addTab(table_widget, "测试结果")
        
        # 可视化标签页
        self.chart_widget = MatplotlibWidget()
        panel.addTab(self.chart_widget, "性能可视化")
        
        # 日志标签页
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        log_toolbar = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_toolbar.addWidget(clear_log_btn)
        log_toolbar.addStretch()
        log_layout.addLayout(log_toolbar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        panel.addTab(log_widget, "测试日志")
        
        return panel
    
    def setup_results_table(self):
        """设置结果表格"""
        headers = ["模型", "场景", "输入", "输出", "耗时(秒)", "Tokens/s", "自评分", "时间戳", "状态"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # 调整列宽
        self.results_table.setColumnWidth(0, 130)  # 模型
        self.results_table.setColumnWidth(1, 130)  # 场景
        self.results_table.setColumnWidth(2, 200)  # 输入
        self.results_table.setColumnWidth(3, 400)  # 输出
        self.results_table.setColumnWidth(4, 80)   # 耗时
        self.results_table.setColumnWidth(5, 80)   # Tokens/s
        self.results_table.setColumnWidth(6, 80)   # 自评分
        self.results_table.setColumnWidth(7, 160)  # 时间戳
        self.results_table.setColumnWidth(8, 80)   # 状态
        
        # 设置表格属性
        self.results_table.setWordWrap(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    
    def init_ollama_connection(self):
        """初始化Ollama连接"""
        try:
            self.ollama_host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
            
            # 测试连接
            resp = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if resp.status_code == 200:
                self.ollama_client = ollama.Client(host=self.ollama_host)
                self.ollama_client.list()  # 验证客户端可用
                self.log_message(f"Ollama连接成功: {self.ollama_host}")
                self.statusBar().showMessage(f"已连接到 Ollama: {self.ollama_host}")
            else:
                raise RuntimeError(f"服务响应异常，状态码: {resp.status_code}")
                
        except Exception as e:
            self.log_message(f"Ollama连接失败: {str(e)}")
            self.statusBar().showMessage("Ollama连接失败")
            self.ollama_client = None
            QMessageBox.warning(self, "连接失败", f"无法连接到Ollama服务:\n{str(e)}")
    
    def load_available_models(self):
        """加载可用的Ollama模型（优先HTTP/Client，失败回退CLI），并以复选框形式呈现"""
        self.model_list.clear()
        self.checked_model_order.clear()
        # 先尝试通过 Client/HTTP 获取
        try:
            if not self.check_ollama_connection():
                raise RuntimeError("Ollama服务不可用")
            models = self.ollama_client.list() if self.ollama_client else ollama.list()
            model_objs = []
            if isinstance(models, dict) and 'models' in models:
                model_objs = models['models']
            elif isinstance(models, list):
                model_objs = models
            else:
                raise RuntimeError(f"未知的模型列表响应格式: {models}")
            names = []
            for m in model_objs:
                name = None
                if isinstance(m, dict):
                    for key in ('name', 'model', 'id', 'title'):
                        if key in m and m[key]:
                            name = m[key]
                            break
                elif isinstance(m, str):
                    name = m
                else:
                    name = str(m)
                if name:
                    names.append(name)
            names = sorted(set(names))
            # 打印并填充
            print("=== 模型列表（HTTP/Client） ===")
            # 成功通过HTTP/Client获取到 names 之后，改为创建复选框项
            for n in names:
                print(n)
                item = QtWidgets.QListWidgetItem(n)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.model_list.addItem(item)
            self.log_message(f"已加载 {len(names)} 个模型（HTTP/Client）")
            return
        except Exception as e_http:
            self.log_message(f"通过HTTP/Client获取模型失败: {str(e_http)}，尝试CLI回退")
            try:
                cmd = "ollama list | sort"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10
                )
                output = result.stdout.strip()
                if result.returncode != 0 or not output:
                    result = subprocess.run(
                        ["ollama", "list"], shell=False, capture_output=True, text=True, timeout=10
                    )
                    output = result.stdout.strip()
                print("=== 模型列表（CLI 原始输出） ===")
                print(output)
                self.log_message("已打印CLI原始输出，请查看控制台。")
                # 解析表格输出
                names = []
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    upper = line.upper()
                    if "NAME" in upper and ("MODIFIED" in upper or "SIZE" in upper):
                        continue
                    first = line.split()[0]
                    names.append(first)
                names = sorted(set(names))
                # CLI 回退路径中，成功解析 names 后同样改为复选框项
                self.model_list.clear()
                for n in names:
                    item = QtWidgets.QListWidgetItem(n)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    item.setCheckState(QtCore.Qt.Unchecked)
                    self.model_list.addItem(item)
                self.log_message(f"已加载 {len(names)} 个模型（CLI）")
            except FileNotFoundError:
                self.log_message("未找到 ollama 可执行文件，请确认已安装并加入 PATH。")
                QMessageBox.warning(self, "错误", "未找到 ollama 可执行文件，请确认已安装并加入 PATH。")
            except Exception as e_cli:
                self.log_message(f"通过CLI获取模型失败: {str(e_cli)}")
                QMessageBox.warning(self, "错误", f"无法获取模型列表: {str(e_cli)}")
    
    def select_all_models(self):
        """选择所有模型"""
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setCheckState(QtCore.Qt.Checked)
    
    def clear_all_models(self):
        """清空所有模型选择"""
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            item.setCheckState(QtCore.Qt.Unchecked)
    
    def on_model_item_changed(self, item):
        """处理模型选择变化"""
        model_name = item.text()
        
        if item.checkState() == QtCore.Qt.Checked:
            if model_name not in self.checked_model_order:
                self.checked_model_order.append(model_name)
                self.log_message(f"选中模型: {model_name}")
        else:
            if model_name in self.checked_model_order:
                self.checked_model_order.remove(model_name)
                self.log_message(f"取消选中模型: {model_name}")
        
        selected_count = len(self.checked_model_order)
        self.statusBar().showMessage(f"已选择 {selected_count} 个模型")
    
    def start_testing(self):
        """开始测试"""
        # 获取选中的模型
        selected_models = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selected_models.append(item.text())
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return
        
        # 获取选中的场景和对应输入
        test_pairs = []
        for i, scenario in enumerate(self.test_scenarios):
            if self.scenario_checkboxes[i].isChecked():
                input_text = self.scenario_inputs[scenario.name].toPlainText().strip()
                if input_text:
                    test_pairs.append((scenario, input_text))
        
        if not test_pairs:
            QMessageBox.warning(self, "警告", "请至少选择一个测试场景并提供输入")
            return
        
        if not self.ollama_client:
            QMessageBox.critical(self, "错误", "Ollama连接未建立，请检查连接")
            return
        
        # 按用户勾选顺序排序模型
        ordered_models = []
        for model in self.checked_model_order:
            if model in selected_models:
                ordered_models.append(model)
        for model in selected_models:
            if model not in ordered_models:
                ordered_models.append(model)
        
        # 创建并启动测试线程
        self.test_worker = ModelTestWorker(ordered_models, test_pairs, self.ollama_client)
        self.test_worker.progress_updated.connect(self.progress_bar.setValue)
        self.test_worker.test_completed.connect(self.on_test_completed)
        self.test_worker.log_updated.connect(self.log_message)
        self.test_worker.status_updated.connect(self.statusBar().showMessage)
        self.test_worker.single_test_completed.connect(self.on_single_test_completed)
        self.test_worker.stream_updated.connect(self.on_stream_updated)
        
        # 更新UI状态
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_message("=" * 50)
        self.log_message(f"开始测试 - 模型: {len(ordered_models)}个, 场景: {len(test_pairs)}个")
        self.log_message("=" * 50)
        
        self.test_worker.start()
    
    def stop_testing(self):
        """停止测试"""
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.stop()
            self.test_worker.wait(3000)  # 等待3秒
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("测试已停止")
        self.log_message("测试已停止")
    
    def on_single_test_completed(self, result: Dict[str, Any]):
        """单个测试完成回调"""
        self.results_data.append(result)
        
        # 添加到表格
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # 创建表格项
        items = [
            QTableWidgetItem(result['model']),
            QTableWidgetItem(result['scenario']),
            QTableWidgetItem(result['input'][:100] + "..." if len(result['input']) > 100 else result['input']),
            QTableWidgetItem(result['output'][:200] + "..." if len(result['output']) > 200 else result['output']),
            QTableWidgetItem(f"{result['duration']:.2f}"),
            QTableWidgetItem(f"{result['tokens_per_second']:.2f}"),
            QTableWidgetItem(f"{result['self_score']:.1f}"),
            QTableWidgetItem(result['timestamp'].split('T')[1][:8]),  # 只显示时间
            QTableWidgetItem(result['status'])
        ]
        
        # 设置状态颜色
        if result['status'] == 'success':
            items[-1].setBackground(QtGui.QColor(144, 238, 144))  # 浅绿色
        else:
            items[-1].setBackground(QtGui.QColor(255, 182, 193))  # 浅红色
        
        # 添加到表格
        for col, item in enumerate(items):
            self.results_table.setItem(row, col, item)
        
        # 调整行高并滚动到底部
        self.results_table.resizeRowToContents(row)
        self.results_table.scrollToBottom()
        
        # 更新统计
        self.update_stats()
        
        # 实时更新可视化
        self.update_visualization()
    
    def on_test_completed(self, data: Dict[str, Any]):
        """测试完成回调"""
        results = data['results']
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        self.statusBar().showMessage(f"测试完成! 成功: {success_count}/{len(results)}")
        
        self.log_message("=" * 50)
        self.log_message(f"测试完成! 总计: {len(results)}个, 成功: {success_count}个")
        self.log_message("=" * 50)
        
        self.update_visualization()
    
    def update_stats(self):
        """更新统计信息"""
        total = len(self.results_data)
        success = sum(1 for r in self.results_data if r['status'] == 'success')
        self.stats_label.setText(f"总计: {total} 个结果 (成功: {success}, 失败: {total-success})")
    
    def update_visualization(self):
        """更新可视化图表"""
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            # 只显示成功的结果
            success_df = df[df['status'] == 'success']
            if not success_df.empty:
                self.chart_widget.plot_performance_comparison(success_df)
    
    def export_to_csv(self):
        """导出结果到CSV"""
        if not self.results_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"ollama_test_results_{timestamp}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", default_filename, "CSV文件 (*.csv)"
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.results_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"结果已导出到:\n{filename}")
                self.log_message(f"结果已导出到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def clear_results(self):
        """清空结果"""
        if not self.results_data:
            return