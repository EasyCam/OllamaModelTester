"""
A tool to use Ollama and LLM to test for some simple local tasks.
"""
import importlib.metadata
import sys
import json
import time
import csv
import threading
from datetime import datetime
from typing import List, Dict, Any

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QTableWidget, 
    QTableWidgetItem, QProgressBar, QTabWidget, QCheckBox,
    QSpinBox, QGroupBox, QSplitter, QFileDialog, QMessageBox,
    QScrollArea
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
    status_updated = Signal(str)  # 新增：用于更新状态栏的信号
    def __init__(self, models: List[str], scenarios: List[TestScenario], test_inputs: List[str], ollama_client):
        super().__init__()
        self.models = models
        self.scenarios = scenarios
        self.test_inputs = test_inputs
        self.results = []
        self.ollama_client = ollama_client
        
    def run(self):
        total_tests = len(self.models) * len(self.scenarios) * len(self.test_inputs)
        completed = 0
        
        for model in self.models:
            for scenario in self.scenarios:
                for test_input in self.test_inputs:
                    try:
                        # 发送状态更新
                        status_msg = f"正在测试: {model} | 场景: {scenario.name} | 进度: {completed+1}/{total_tests}"
                        self.status_updated.emit(status_msg)
                        self.log_updated.emit(f"测试模型: {model}, 场景: {scenario.name}")
                        result = self.run_single_test(model, scenario, test_input)
                        self.results.append(result)
                        
                        completed += 1
                        progress = int((completed / total_tests) * 100)
                        self.progress_updated.emit(progress)
                        
                    except Exception as e:
                        self.log_updated.emit(f"错误: {str(e)}")
                        
        self.test_completed.emit({'results': self.results})
    
    def run_single_test(self, model: str, scenario: TestScenario, test_input: str) -> Dict[str, Any]:
        """运行单个测试"""
        prompt = scenario.prompt_template.format(input=test_input)
        
        start_time = time.time()
        
        try:
            # 调用Ollama模型（优先使用传入的客户端）
            if self.ollama_client:
                response = self.ollama_client.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                )
            else:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                )
            options={'verbose': True}
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 安全地获取输出内容
            output = ""
            if isinstance(response, dict):
                if 'message' in response and isinstance(response['message'], dict):
                    output = response['message'].get('content', '未获取到响应内容')
                else:
                    output = str(response)
            else:
                output = str(response)
            
            # 计算tokens/s (估算)
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
            end_time = time.time()
            duration = end_time - start_time
            
            error_msg = f"测试失败: {str(e)}"
            self.log_updated.emit(error_msg)
            
            return {
                'model': model,
                'scenario': scenario.name,
                'input': test_input,
                'output': error_msg,
                'duration': duration,
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
        
        请只返回一个1-10之间的数字分数。
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
            # 尝试提取数字
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', score_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 1), 10)  # 确保在1-10范围内
            return 5.0  # 默认分数
        except:
            return 5.0


class MatplotlibWidget(QWidget):
    """Matplotlib图表组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_performance_comparison(self, results_df: pd.DataFrame):
        """绘制性能对比图"""
        self.figure.clear()
        
        # 创建子图
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        
        # Tokens/s 对比
        model_performance = results_df.groupby('model')['tokens_per_second'].mean()
        ax1.bar(model_performance.index, model_performance.values)
        ax1.set_title('平均 Tokens/s')
        ax1.tick_params(axis='x', rotation=45)
        
        # 响应时间对比
        model_duration = results_df.groupby('model')['duration'].mean()
        ax2.bar(model_duration.index, model_duration.values)
        ax2.set_title('平均响应时间 (秒)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 自评分对比
        model_scores = results_df.groupby('model')['self_score'].mean()
        ax3.bar(model_scores.index, model_scores.values)
        ax3.set_title('平均自评分')
        ax3.tick_params(axis='x', rotation=45)
        
        # 场景表现对比
        scenario_scores = results_df.groupby('scenario')['self_score'].mean()
        ax4.bar(scenario_scores.index, scenario_scores.values)
        ax4.set_title('各场景平均得分')
        ax4.tick_params(axis='x', rotation=45)
        
        self.figure.tight_layout()
        self.canvas.draw()


class OllamaModelTester(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.test_worker = None
        self.results_data = []
        self.ollama_client = None
        self.checked_model_order = []  # 记录用户勾选顺序的模型名
        self.init_test_scenarios()
        self.init_ui()
        self.init_ollama_connection()
        self.load_available_models()
    
    def init_ollama_connection(self):
        """初始化Ollama连接"""
        try:
            # 从环境变量读取主机，默认 127.0.0.1 避免 DNS 问题
            self.ollama_host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
            # 先探测服务是否可用
            resp = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"服务未就绪，状态码: {resp.status_code}")
            # 创建客户端
            self.ollama_client = ollama.Client(host=self.ollama_host)
            # 再次用 list() 验证
            self.ollama_client.list()
            self.log_message(f"Ollama连接成功: {self.ollama_host}")
        except Exception as e:
            self.log_message(f"Ollama连接失败: {str(e)}")
            self.ollama_client = None
    
    def check_ollama_connection(self) -> bool:
        """检查Ollama连接状态"""
        try:
            if self.ollama_client is None:
                self.init_ollama_connection()
            # 再次确认服务可达
            host = getattr(self, 'ollama_host', 'http://127.0.0.1:11434').rstrip('/')
            resp = requests.get(f"{host}/api/version", timeout=5)
            if resp.status_code != 200:
                self.log_message(f"Ollama版本检查失败，状态码: {resp.status_code}")
                return False
            # 客户端 list 校验
            if self.ollama_client:
                self.ollama_client.list()
                return True
            return False
        except Exception as e:
            self.log_message(f"Ollama连接检查失败: {str(e)}")
            return False
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        export_action = QtGui.QAction('导出结果', self)
        export_action.triggered.connect(self.export_to_csv)
        file_menu.addAction(export_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        refresh_action = QtGui.QAction('刷新模型', self)
        refresh_action.triggered.connect(self.load_available_models)
        tools_menu.addAction(refresh_action)
    
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
    
    def init_test_scenarios(self):
        """初始化测试场景"""
        self.test_scenarios = [
            TestScenario(
                "自然语言转Python",
                "将自然语言描述转换为Python代码",
                "请将以下自然语言描述转换为Python代码:\n{input}",
                "代码正确性、可读性、效率"
            ),
            TestScenario(
                "中英互译",
                "中英文双向翻译",
                "请翻译以下文本:\n{input}",
                "翻译准确性、流畅性、语言地道性"
            ),
            TestScenario(
                "代码解释",
                "解释代码功能和逻辑",
                "请解释以下代码的功能和逻辑:\n{input}",
                "解释准确性、清晰度、完整性"
            ),
            TestScenario(
                "问题解答",
                "回答技术问题",
                "请回答以下问题:\n{input}",
                "答案准确性、详细程度、实用性"
            )
        ]

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Ollama 模型测试工具")
        self.setGeometry(100, 100, 1400, 900)
        
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
        splitter.setSizes([400, 1000])
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 创建菜单栏
        self.create_menu_bar()
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_list = QtWidgets.QListWidget()
        self.model_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.model_list.itemChanged.connect(self.on_model_item_changed)
        model_layout.addWidget(self.model_list)
        
        refresh_btn = QPushButton("刷新模型列表")
        refresh_btn.clicked.connect(self.load_available_models)
        model_layout.addWidget(refresh_btn)
        
        debug_btn = QPushButton("调试连接")
        debug_btn.clicked.connect(self.check_ollama_connection)
        model_layout.addWidget(debug_btn)
        
        layout.addWidget(model_group)
        
        # 测试场景组
        scenario_group = QGroupBox("测试场景")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_checkboxes = []
        for scenario in self.test_scenarios:
            checkbox = QCheckBox(scenario.name)
            checkbox.setChecked(True)
            self.scenario_checkboxes.append(checkbox)
            scenario_layout.addWidget(checkbox)
        
        layout.addWidget(scenario_group)
        
        # 测试输入组
        input_group = QGroupBox("测试输入")
        input_layout = QVBoxLayout(input_group)
        
        self.test_input = QTextEdit()
        self.test_input.setPlainText(
            "计算斐波那契数列的第10项\n"
            "Hello, how are you today?\n"
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)\n"
            "什么是辉长结构？"
        )
        input_layout.addWidget(self.test_input)
        
        layout.addWidget(input_group)
        
        # 控制按钮组
        button_group = QGroupBox("测试控制")
        button_layout = QVBoxLayout(button_group)
        
        self.start_btn = QPushButton("开始测试")
        self.start_btn.clicked.connect(self.start_testing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止测试")
        self.stop_btn.clicked.connect(self.stop_testing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        button_layout.addWidget(self.progress_bar)
        
        layout.addWidget(button_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def create_results_panel(self) -> QWidget:
        """创建结果面板"""
        panel = QTabWidget()
        
        # 结果表格标签页
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        # 表格工具栏
        table_toolbar = QHBoxLayout()
        
        export_btn = QPushButton("导出CSV")
        export_btn.clicked.connect(self.export_to_csv)
        table_toolbar.addWidget(export_btn)
        
        clear_btn = QPushButton("清空结果")
        clear_btn.clicked.connect(self.clear_results)
        table_toolbar.addWidget(clear_btn)
        
        table_toolbar.addStretch()
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
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        panel.addTab(self.log_text, "测试日志")
        
        return panel
    
    def setup_results_table(self):
        """设置结果表格"""
        headers = ["模型", "场景", "输入", "输出", "耗时(秒)", "Tokens/s", "自评分", "时间戳"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # 设置列宽
        self.results_table.setColumnWidth(0, 120)
        self.results_table.setColumnWidth(1, 120)
        self.results_table.setColumnWidth(2, 200)
        self.results_table.setColumnWidth(3, 300)
        self.results_table.setColumnWidth(4, 80)
        self.results_table.setColumnWidth(5, 80)
        self.results_table.setColumnWidth(6, 80)
        self.results_table.setColumnWidth(7, 150)
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        export_action = QtGui.QAction('导出结果', self)
        export_action.triggered.connect(self.export_to_csv)
        file_menu.addAction(export_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        refresh_action = QtGui.QAction('刷新模型', self)
        refresh_action.triggered.connect(self.load_available_models)
        tools_menu.addAction(refresh_action)

    def start_testing(self):
        """开始测试"""
        # 检查Ollama连接
        if not self.check_ollama_connection():
            QMessageBox.warning(self, "连接错误", "无法连接到Ollama服务，请检查服务状态")
            return
        
        # 使用勾选顺序
        selected_models = list(self.checked_model_order)
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少勾选一个模型")
            return
        
        # 获取选中的场景（保持原逻辑）
        selected_scenarios = []
        for i, checkbox in enumerate(self.scenario_checkboxes):
            if checkbox.isChecked():
                selected_scenarios.append(self.test_scenarios[i])
        
        if not selected_scenarios:
            QMessageBox.warning(self, "警告", "请至少选择一个测试场景")
            return
        
        # 获取测试输入（保持原逻辑）
        test_inputs = [line.strip() for line in self.test_input.toPlainText().split('\n') if line.strip()]
        if not test_inputs:
            QMessageBox.warning(self, "警告", "请输入测试内容")
            return
        
        # 启动测试线程（保持原逻辑，但模型顺序来自勾选顺序）
        self.test_worker = ModelTestWorker(selected_models, selected_scenarios, test_inputs, self.ollama_client)
        self.test_worker.progress_updated.connect(self.update_progress)
        self.test_worker.test_completed.connect(self.on_test_completed)
        self.test_worker.log_updated.connect(self.log_message)
        self.test_worker.status_updated.connect(self.update_status)  # 新增：连接状态更新信号
        
        self.test_worker.start()
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("准备开始测试...")
        self.log_message("开始批量测试...")
    

    def on_model_item_changed(self, item: QtWidgets.QListWidgetItem):
        """当用户勾选/取消勾选模型时，维护勾选顺序"""
        name = item.text()
        if item.checkState() == QtCore.Qt.Checked:
            if name in self.checked_model_order:
                self.checked_model_order.remove(name)
            self.checked_model_order.append(name)
        else:
            if name in self.checked_model_order:
                self.checked_model_order.remove(name)
    def stop_testing(self):
        """停止测试"""
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.terminate()
            self.test_worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("测试已停止")
        self.log_message("测试已停止")
    
    def update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_status(self, status_message: str):
        """更新状态栏显示"""
        self.statusBar().showMessage(status_message)

    def on_test_completed(self, data: Dict[str, Any]):
        """测试完成回调"""
        results = data['results']
        self.results_data.extend(results)
        
        # 更新结果表格
        self.update_results_table(results)
        
        # 更新可视化
        self.update_visualization()
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage(f"测试完成，共 {len(results)} 个结果")
        
        self.log_message(f"测试完成，共生成 {len(results)} 个结果")
    
    def update_results_table(self, results: List[Dict[str, Any]]):
        """更新结果表格"""
        for result in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            self.results_table.setItem(row, 0, QTableWidgetItem(result['model']))
            self.results_table.setItem(row, 1, QTableWidgetItem(result['scenario']))
            self.results_table.setItem(row, 2, QTableWidgetItem(result['input'][:50] + "..."))
            self.results_table.setItem(row, 3, QTableWidgetItem(result['output'][:100] + "..."))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{result['duration']:.2f}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{result['tokens_per_second']:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{result['self_score']:.1f}"))
            self.results_table.setItem(row, 7, QTableWidgetItem(result['timestamp']))
    
    def update_visualization(self):
        """更新可视化图表"""
        if not self.results_data:
            return
        
        df = pd.DataFrame(self.results_data)
        self.chart_widget.plot_performance_comparison(df)
    
    def export_to_csv(self):
        """导出结果到CSV"""
        if not self.results_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", f"ollama_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.results_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"结果已导出到: {filename}")
                self.log_message(f"结果已导出到: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def clear_results(self):
        """清空结果"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有结果吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.results_data.clear()
            self.results_table.setRowCount(0)
            self.chart_widget.figure.clear()
            self.chart_widget.canvas.draw()
            self.log_message("结果已清空")
    
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    
    # Handle both briefcase and direct python execution
    try:
        if app_module:
            # Retrieve the app's metadata (briefcase mode)
            metadata = importlib.metadata.metadata(app_module)
            app_name = metadata["Formal-Name"]
        else:
            # Direct python execution mode
            app_name = "OllamaModelTester"
    except (ValueError, importlib.metadata.PackageNotFoundError):
        # Fallback for direct execution
        app_name = "OllamaModelTester"

    QtWidgets.QApplication.setApplicationName(app_name)

    app = QtWidgets.QApplication(sys.argv)
    main_window = OllamaModelTester()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

