import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, Scale, Menu, filedialog
import threading
from openai import OpenAI
import time
import jsonschema
import json
import base64
import os
import re
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import PyPDF2  # 确保已安装 PyPDF2
import jieba  # 中文分词库
from rich.console import Console
from rich.markdown import Markdown
import sys
import os
import yaml
import fitz  # PyMuPDF
from docx import Document
import requests
import json
from urllib.parse import urljoin, urlencode
# Python 代码
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件
import os
model_api_key = os.getenv("OPENAI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
search_api_key = os.getenv("SEARCH_API_KEY")
# 工具配置文件路径
TOOL_CONFIG_FILE = "tool_config.yaml"

# 定义输入数据的JSON Schema
console = Console()
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
            "pattern": "^[\\u4e00-\\u9fa5a-zA-Z0-9\\s,.!?;:，。！？；：'\"()\\[\\]\\-+=_<>/\\\\]*$"
        },
        "model": {
            "type": "string",
            "enum": ["Pro/deepseek-ai/DeepSeek-R1", "Wan-AI/Wan2.1-T2V-14B-Turbo", "Kwai-Kolors/Kolors","Qwen/QVQ-72B-Preview"]
        },
        "temperature": {
            "type": "number",
            "minimum": 0,
            "maximum": 2
        }
    },
    "required": ["question"]
}

# 定义输出数据的JSON Schema
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error", "loading", "tool_required"]
        },
        "content": {
            "type": "string",
            "minLength": 1
        },
        "timestamp": {
            "type": "number"
        },
        "model": {
            "type": "string"
        },
        "rag_context": {
            "type": "string"
        },
        "tool_used": {
            "type": "string",
            "enum": ["none", "file", "weather", "search"]
        }
    },
    "required": ["status", "content", "timestamp", "model", "tool_used"]
}

class ChatApp:
    def __init__(self, root:tk.Tk):
        self.root = root
        self.root.title("AI聊天助手 - 合并功能版")
        self.root.geometry("900x800")  # 增加高度以容纳新组件
        # 使用更中性的背景色
        self.root.configure(bg="#F0F0F0")
        self.root.resizable(True, True)
        
        # 设置字体以支持中文显示
        self.font_family = ("Microsoft YaHei UI", "SimHei", "Arial")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=model_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.current_image = None
        self.image_preview = None
        self.image_path = None
        # 聊天历史和当前状态
        self.chat_history = []  # 完整对话历史
        self.current_question = ""
        self.current_model = "Pro/deepseek-ai/DeepSeek-R1"
        self.current_temperature = 0.7
        self.is_waiting_response = False
        self.max_history_length = 5  # 保留的最大对话轮数
        
        # RAG知识库
        self.knowledge_base = []  # 存储知识库文档
        self.vectorizer = TfidfVectorizer()  # 用于文本向量化
        self.tfidf_matrix = None  # 存储向量化后的文档
        self.doc_embeddings = {}  # 存储文档向量
        
        # 添加文件列表状态
        self.file_list_var = tk.StringVar(value="")
        
        # 初始化jieba分词器
        jieba.initialize()  # 初始化分词器
        self.stopwords = self._load_stopwords()  # 加载停用词
        self.tool_config = self._load_tool_config()
        
        # 工具开关
        self.file_enabled = self.tool_config.get("file", {}).get("enabled", True)
        self.weather_enabled = self.tool_config.get("weather", {}).get("enabled", True)
        self.search_enabled = self.tool_config.get("search", {}).get("enabled", True)
        self.weather_api_key = self.tool_config.get("weather", {}).get("api_key", weather_api_key)
        self.search_api_key = self.tool_config.get("search", {}).get("api_key", search_api_key)
        self.search_provider = self.tool_config.get("search", {}).get("provider", "google")
        print(f"ChatApp init: file_enabled={self.file_enabled}, weather_enabled={self.weather_enabled}, search_enabled={self.search_enabled}")        
        
        # 创建UI组件
        self._create_ui()
        
        # 初始化文件列表
        self.refresh_file_list()
        
    def _load_tool_config(self):
        """加载工具配置（新增搜索配置）"""
        try:
            if os.path.exists(TOOL_CONFIG_FILE):
                print("存在，已加载工具配置文件")
                with open(TOOL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    result = yaml.safe_load(f)
                    if result:
                        print("yaml.safe_load:成功")
                        return result
                    else:
                        print(f"{os.getcwd() + '\\' + TOOL_CONFIG_FILE + "为空"} 创建默认工具配置文件")
                        default_config = {
                            "file": {"enabled": True, "allowed_dirs": [os.getcwd()]},
                            "weather": {"enabled": True, "api_key": weather_api_key},
                            "search": {"enabled": True, "api_key": search_api_key}
                        }                        
                        with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                            yaml.dump(default_config, f)
                        return default_config
            else:
                # 默认配置（文件读取+天气查询+网络搜索）
                print("暂不存在，已创建默认工具配置文件")
                default_config = {
                    "file": {"enabled": True, "allowed_dirs": [os.getcwd()]},
                    "weather": {"enabled": True, "api_key": weather_api_key},
                    "search": {"enabled": True, "api_key": search_api_key, "provider": "google"}
                }
                with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f)
                return default_config
        except Exception as e:
            messagebox.showerror("配置错误", f"加载工具配置失败: {str(e)}")
            return {
                "file": {"enabled": True, "allowed_dirs": [os.getcwd()]},
                "weather": {"enabled": True, "api_key": weather_api_key},
                "search": {"enabled": True, "api_key": search_api_key, "provider": "google"}
            }
    def _load_stopwords(self):
        """加载中文停用词表"""
        stopwords = set()
        try:
            # 尝试加载本地停用词表
            with open("stopwords.txt", "r", encoding="utf-8") as f:
                for line in f:
                    stopwords.add(line.strip())
        except:
            # 内置一个基本的中文停用词表
            basic_stopwords = {
                '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', 
                '在', '一个', '有', '我', '他', '她', '它', '我们', '他们', 
                '这', '那', '这些', '那些', '为', '为了', '因为', '所以', 
                '但', '虽然', '如果', '然后', '当', '只', '仅', '也', '还',
                '又', '或', '或者', '并且', '而且', '甚至', '例如', '比如',
                '如', '像', '什么', '怎么', '如何', '为什么', '哪里', '谁',
                '何时', '多少', '非常', '很', '太', '更', '最', '比较', 
                '些', '个', '种', '项', '条', '点', '方面', '问题', '答案'
            }
            stopwords = basic_stopwords
        
        return stopwords
    
    def chinese_tokenizer(self, text):
        """中文分词函数，用于TF-IDF向量化"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 过滤停用词和单字
        filtered_words = [
            word for word in words 
            if (word not in self.stopwords) and (len(word) > 1)
        ]
        return filtered_words
    
    def _create_ui(self):
        # 创建主Canvas用于滚动
        self.canvas = tk.Canvas(self.root, bg="#E0E0E0", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加垂直滚动条
        self.scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 在Canvas中创建可滚动的框架
        self.main_frame = tk.Frame(self.canvas, bg="#E0E0E0")
        self.main_frame_id = self.canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)
        
        # 绑定配置事件，用于更新滚动区域
        self.main_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # 顶部信息栏
        info_frame = tk.Frame(self.main_frame, bg="#333333")
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        
        # 顶部信息栏（以下代码保持不变，仅调整父容器为main_frame）
        info_frame = tk.Frame(self.main_frame, bg="#333333")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        main_frame = self.main_frame
        
        ttk.Label(
            info_frame, text="AI聊天助手 - 功能合并版",
            background="#333333", foreground="#FFFFFF",
            font=(self.font_family[0], 14, "bold")
        ).pack(side=tk.LEFT, padx=10, pady=5)
        # 工具开关框架
        tool_frame = tk.Frame(main_frame, bg="#E0E0E0", bd=1, relief=tk.RAISED)
        tool_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        ttk.Label(
            tool_frame, text="可用工具:",
            background="#E0E0E0", foreground="#333333",
            font=(self.font_family[0], 10, "bold")
        ).pack(side=tk.LEFT, padx=(10, 5), pady=5)
        
        self.file_var = tk.BooleanVar(value=self.file_enabled)
        ttk.Checkbutton(
            tool_frame, text="文件读取", variable=self.file_var,
            command=lambda: self._toggle_file_tool(self.file_var.get())
        ).pack(side=tk.LEFT, padx=(0, 5), pady=5)
        
        self.weather_var = tk.BooleanVar(value=self.weather_enabled)
        ttk.Checkbutton(
            tool_frame, text="天气查询", variable=self.weather_var,
            command=lambda: self._toggle_weather_tool(self.weather_var.get())
        ).pack(side=tk.LEFT, padx=(0, 5), pady=5)
        
        self.search_var = tk.BooleanVar(value=self.search_enabled)
        ttk.Checkbutton(
            tool_frame, text="网络搜索", variable=self.search_var,
            command=lambda: self._toggle_search_tool(self.search_var.get())
        ).pack(side=tk.LEFT, padx=(0, 5), pady=5)
                
        # 创建菜单栏
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        tool_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tool_menu)
        tool_menu.add_command(label="配置文件读取", command=self.run_file_config_gui)
        tool_menu.add_command(label="配置天气查询", command=self.run_weather_config_gui)
        tool_menu.add_command(label="配置网络搜索", command=self.run_search_config_gui)        
        # 添加"记忆"菜单
        memory_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="记忆管理", menu=memory_menu)
        memory_menu.add_command(label="清除记忆", command=self.clear_memory)
        memory_menu.add_command(label="保存记忆", command=self.save_memory)
        memory_menu.add_command(label="加载记忆", command=self.load_memory_dialog)
        memory_menu.add_command(label="查看保存的记忆", command=self.view_saved_memories)
        
        # 添加"知识库"菜单
        knowledge_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="知识库管理", menu=knowledge_menu)
        knowledge_menu.add_command(label="加载知识文档", command=self.load_knowledge_dialog)
        knowledge_menu.add_command(label="清空知识库", command=self.clear_knowledge_base)
        knowledge_menu.add_command(label="查看知识库", command=self.view_knowledge_base)
        
        
        
        # 模型选择框架
        model_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(
            model_frame, text="选择模型:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 10, "bold")
        ).pack(side=tk.LEFT, padx=(10, 10), pady=5)
        
        self.model_combo = ttk.Combobox(
            model_frame,
            values=["Pro/deepseek-ai/DeepSeek-R1", "Wan-AI/Wan2.1-T2V-14B-Turbo", "Kwai-Kolors/Kolors","Qwen/QVQ-72B-Preview"],
            state="readonly",
            width=20
        )
        self.model_combo.set(self.current_model)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        # 温度设置框架
        temp_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        temp_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(
            temp_frame, text="Temperature:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 10, "bold")
        ).pack(side=tk.LEFT, padx=(10, 10), pady=5)
        
        self.temp_scale = Scale(
            temp_frame, from_=0, to=2, orient=tk.HORIZONTAL,
            length=200, resolution=0.1,
            command=self._on_temperature_change,
            bg="#FFFFFF", fg="#333333",
            highlightbackground="#E0E0E0"
        )
        self.temp_scale.set(self.current_temperature)
        self.temp_scale.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        
        self.temp_label = ttk.Label(
            temp_frame, text=f"{self.current_temperature:.1f}",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 10)
        )
        self.temp_label.pack(side=tk.LEFT, pady=5)
        image_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        image_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(
            image_frame, text="图像上传:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 10, "bold")
        ).pack(side=tk.LEFT, padx=(10, 10), pady=5)
        
        self.upload_button = ttk.Button(
            image_frame, text="选择图片", command=self.upload_image,
            style="TButton"
        )
        self.upload_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        
        self.clear_image_button = ttk.Button(
            image_frame, text="清除图片", command=self.clear_image,
            style="TButton"
        )
        self.clear_image_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)

        # 图像预览区域
        self.image_preview_label = tk.Label(
            image_frame, bg="#FFFFFF", 
            text="(无图片)", font=(self.font_family[0], 9)
        )
        self.image_preview_label.pack(side=tk.LEFT, padx=(10, 10), pady=5)
        # RAG设置框架
        rag_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        rag_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # RAG开关
        self.rag_enabled = tk.BooleanVar(value=True)
        rag_switch = ttk.Checkbutton(
            rag_frame, text="启用RAG增强",
            variable=self.rag_enabled,
            style="Switch.TCheckbutton"
        )
        rag_switch.pack(side=tk.LEFT, padx=10, pady=5)
        
        # RAG上下文长度
        ttk.Label(
            rag_frame, text="上下文片段:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 9)
        ).pack(side=tk.LEFT, padx=(10, 0), pady=5)
        
        self.rag_context_var = tk.IntVar(value=3)
        rag_context_combo = ttk.Combobox(
            rag_frame,
            textvariable=self.rag_context_var,
            values=[1, 2, 3, 4, 5],
            state="readonly",
            width=3
        )
        rag_context_combo.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        
        # 知识库状态
        self.knowledge_status = tk.StringVar(value="知识库: 0 个文档")
        ttk.Label(
            rag_frame, textvariable=self.knowledge_status,
            background="#FFFFFF", foreground="#666666",
            font=(self.font_family[0], 9)
        ).pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 记忆状态显示框架
        memory_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        memory_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 左侧：记忆状态标签
        self.memory_status = tk.StringVar(value=f"记忆状态: 保存了 {len(self.chat_history)//2} 轮对话")
        ttk.Label(
            memory_frame, textvariable=self.memory_status,
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 9)
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # 右侧：记忆管理按钮
        btn_frame = tk.Frame(memory_frame, bg="#FFFFFF")
        btn_frame.pack(side=tk.RIGHT, padx=(0, 10))
        
        # 保存记忆按钮
        self.save_btn = ttk.Button(
            btn_frame, text="保存记忆", width=8,
            command=self.save_memory,
            style="Memory.TButton"
        )
        self.save_btn.pack(side=tk.LEFT, padx=3)
        
        # 加载记忆按钮
        self.load_btn = ttk.Button(
            btn_frame, text="加载记忆", width=8,
            command=self.load_memory_dialog,
            style="Memory.TButton"
        )
        self.load_btn.pack(side=tk.LEFT, padx=3)
        
        # 清除记忆按钮
        self.clear_btn = ttk.Button(
            btn_frame, text="清除记忆", width=8,
            command=self.clear_memory,
            style="Accent.TButton"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=3)
        
        # 文件列表区域
        file_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        file_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 文件列表标签
        ttk.Label(
            file_frame, text="记忆文件列表:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 9, "bold")
        ).pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        
        # 文件列表显示区域（带滚动条）
        file_list_frame = tk.Frame(file_frame, bg="#FFFFFF")
        file_list_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # 滚动条
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 文件列表框
        self.file_listbox = tk.Listbox(
            file_list_frame, 
            height=3,  # 显示3行文件
            yscrollcommand=file_scrollbar.set,
            bg="#FFFFFF", fg="#333333",
            selectbackground="#4A86E8", selectforeground="#FFFFFF",
            font=(self.font_family[0], 9),
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.config(command=self.file_listbox.yview)
        
        # 绑定双击事件
        self.file_listbox.bind("<Double-Button-1>", self._on_file_double_click)
        
        # 刷新按钮
        refresh_frame = tk.Frame(file_frame, bg="#FFFFFF")
        refresh_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.refresh_btn = ttk.Button(
            refresh_frame, text="刷新列表", width=10,
            command=self.refresh_file_list,
            style="Memory.TButton"
        )
        self.refresh_btn.pack(side=tk.LEFT, pady=(0, 5))
        
        # 状态信息
        self.file_status = ttk.Label(
            refresh_frame, textvariable=self.file_list_var,
            background="#FFFFFF", foreground="#666666",
            font=(self.font_family[0], 8)
        )
        self.file_status.pack(side=tk.LEFT, padx=10)
        
        # 聊天历史显示区域 - 使用更清晰的背景和文本颜色对比
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, height=15,  # 高度减少以容纳新组件
            bg="#FFFFFF", fg="#333333", insertbackground="#333333",
            font=(self.font_family[0], 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.tag_config("user_header", foreground="#007ACC", font=(self.font_family[0], 10, "bold"))
        self.chat_display.tag_config("user_message", foreground="#333333")
        self.chat_display.tag_config("ai_header", foreground="#D83B01", font=(self.font_family[0], 10, "bold"))
        self.chat_display.tag_config("ai_message", foreground="#333333")
        self.chat_display.tag_config("tool_message", foreground="#006400", font=(self.font_family[0], 10, "italic"))
        self.chat_display.tag_config("error_message", foreground="#D13438", font=(self.font_family[0], 10, "bold"))          
        # RAG上下文显示区域
        rag_context_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        rag_context_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(
            rag_context_frame, text="RAG上下文:",
            background="#FFFFFF", foreground="#333333",
            font=(self.font_family[0], 9, "bold")
        ).pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        
        self.rag_context_display = scrolledtext.ScrolledText(
            rag_context_frame, wrap=tk.WORD, height=4,
            bg="#F8F8F8", fg="#555555", insertbackground="#555555",
            font=(self.font_family[0], 9)
        )
        self.rag_context_display.pack(fill=tk.BOTH, padx=10, pady=(0, 5))
        self.rag_context_display.config(state=tk.DISABLED)
        
        # 输入区域框架
        input_frame = tk.Frame(main_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 输入框
        self.input_box = scrolledtext.ScrolledText(
            input_frame, wrap=tk.WORD, height=4,
            bg="#F9F9F9", fg="#333333", insertbackground="#333333",
            font=(self.font_family[0], 10)
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.input_box.bind("<Return>", self._on_enter_key)
        self.input_box.bind("<Shift-Return>", lambda event: "\n")
        self.input_box.focus_set()
        
        # 发送按钮 - 使用更明显的强调色
        self.send_button = ttk.Button(
            input_frame, text="发送", command=self.send_message,
            style="Accent.TButton"
        )
        self.send_button.pack(side=tk.LEFT, padx=(0, 10), pady=10)

        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_bar = ttk.Label(
            main_frame, textvariable=self.status_var,
            background="#333333", foreground="#FFFFFF",
            font=(self.font_family[0], 9)
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 10))
        
        # 配置主题样式
        self._setup_styles()
    def _on_frame_configure(self, event):
        """当滚动框架大小改变时，更新Canvas的滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_canvas_configure(self, event):
        """当Canvas大小改变时，调整滚动框架的宽度"""
        self.canvas.itemconfig(self.main_frame_id, width=event.width)    
        
        
    def _toggle_file_tool(self, enabled):
        """切换文件读取工具开关"""
        self.file_enabled = enabled
        print(f"toggle file tool:切换为: {enabled}")
        self.tool_config["file"]["enabled"] = enabled
            
        with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self.tool_config, f)
            
        self.status_var.set(f"已{'启用' if enabled else '禁用'}文件读取工具")
        
    def _toggle_weather_tool(self, enabled):
        """切换天气查询工具开关"""
        self.weather_enabled = enabled
        print(f"toggle weather tool:切换为: {enabled}")
        self.tool_config["weather"]["enabled"] = enabled
            
        with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self.tool_config, f)
            
        self.status_var.set(f"已{'启用' if enabled else '禁用'}天气查询工具")
        
    def _toggle_search_tool(self, enabled):
        """切换网络搜索工具开关"""
        self.search_enabled = enabled
        print(f"toggle search tool:切换为: {enabled}")
        self.tool_config["search"]["enabled"] = enabled
            
        with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self.tool_config, f)
            
        self.status_var.set(f"已{'启用' if enabled else '禁用'}网络搜索工具")    
    def _setup_styles(self):
        """配置ttk主题样式"""
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        style.configure(
            "TButton",
            background="#4A86E8",
            foreground="#FFFFFF",
            borderwidth=1,
            relief=tk.FLAT,
            font=(self.font_family[0], 10, "bold"),
            padding=(10, 5)
        )
        style.map(
            "TButton",
            background=[("active", "#3A76D8"), ("disabled", "#CCCCCC")],
            foreground=[("disabled", "#999999")]
        )
        
        style.configure(
            "Accent.TButton",
            background="#E64A19",
            foreground="#FFFFFF"
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#D63A09"), ("disabled", "#CCCCCC")]
        )
        
        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background="#E0E0E0",
            darkcolor="#D0D0D0",
            lightcolor="#D0D0D0",
            troughcolor="#F0F0F0",
            bordercolor="#D0D0D0",
            arrowcolor="#666666"
        )
        
        # 添加记忆按钮样式
        style.configure(
            "Memory.TButton",
            background="#4A86E8",
            foreground="#FFFFFF",
            borderwidth=1,
            relief=tk.FLAT,
            font=(self.font_family[0], 9),
            padding=(5, 2)
        )
        style.map(
            "Memory.TButton",
            background=[("active", "#3A76D8"), ("disabled", "#CCCCCC")],
            foreground=[("disabled", "#999999")]
        )
        
        # 开关按钮样式
        style.configure(
            "Switch.TCheckbutton",
            font=(self.font_family[0], 9, "bold"),
            foreground="#333333"
        )
        
    def refresh_file_list(self):
        """刷新文件列表"""
        # 获取当前目录下所有记忆文件
        memory_files = [f for f in os.listdir(".") 
                       if f.startswith("chat_") and f.endswith(".json")]
        
        # 按修改时间排序（最新的在前）
        memory_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 更新列表框
        self.file_listbox.delete(0, tk.END)
        for file in memory_files:
            # 显示文件名和修改时间
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            display_text = f"{file} (修改于: {mod_time.strftime('%Y-%m-%d %H:%M')})"
            self.file_listbox.insert(tk.END, display_text)
            
        # 更新状态信息
        file_count = len(memory_files)
        self.file_list_var.set(f"找到 {file_count} 个记忆文件")
        
    def _on_file_double_click(self, event):
        """处理文件双击事件"""
        # 获取选中的文件索引
        selection = self.file_listbox.curselection()
        if not selection:
            return
            
        # 获取实际文件名（去掉时间信息）
        display_text = self.file_listbox.get(selection[0])
        filename = display_text.split(" (修改于:")[0]
        
        # 加载记忆
        self.load_memory(filename)
        
    def _on_model_selected(self, event):
        """模型选择变更处理"""
        self.current_model = self.model_combo.get()
        
        # 根据模型类型启用/禁用图像上传功能
        if self.current_model == "Qwen/QVQ-72B-Preview":
            self.upload_button.config(state=tk.NORMAL)
            self.clear_image_button.config(state=tk.NORMAL)
            if not self.current_image:
                self.status_var.set(f"已选择模型: {self.current_model} (请上传图片)")
        else:
            self.upload_button.config(state=tk.DISABLED)
            self.clear_image_button.config(state=tk.DISABLED)
            self.current_image = None  # 明确设置为None
            self.image_preview_label.configure(text="(无图片)")
            self.image_preview_label.image = None  # 清除图片引用
        
        self.status_var.set(f"已选择模型: {self.current_model}")
        
    def _on_temperature_change(self, value):
        """处理温度滑块变更"""
        self.current_temperature = float(value)
        self.temp_label.config(text=f"{self.current_temperature:.1f}")
        
    def _on_enter_key(self, event):
        """处理Enter键事件"""
        if not event.state & 0x1:
            self.send_message()
            return "break"
        return None
    def upload_image(self):
        """上传并预览图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not file_path:
            return
            
        try:
            # 检查文件大小（限制在5MB以内）
            if os.path.getsize(file_path) > 5 * 1024 * 1024:
                messagebox.showerror("错误", "图片大小不能超过5MB")
                return
                
            # 加载并预览图像
            img = Image.open(file_path)
            img.thumbnail((150, 150))  # 调整缩略图尺寸
            
            # 转换为Base64编码
            with open(file_path, "rb") as image_file:
                self.current_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # 保存图像路径用于显示
            self.image_path = file_path
            
            # 显示预览
            photo = ImageTk.PhotoImage(img)
            self.image_preview_label.configure(image=photo)
            self.image_preview_label.image = photo  # 保持引用
            self.image_preview_label.configure(text="")
            
            self.status_var.set(f"已加载图片: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            self.current_image = None
            self.image_path = None

    def clear_image(self):
        """清除已上传的图像"""
        self.current_image = None
        self.image_path = None
        self.image_preview_label.configure(image=None)
        self.image_preview_label.configure(text="(无图片)")
        self.status_var.set("已清除图片")

   
    def format_output(self, content, status="success", rag_context="",tool_used="none"):
        """按规范格式化输出数据"""
        return {
            "status": status,
            "content": content,
            "timestamp": time.time(),
            "model": self.current_model,
            "rag_context": rag_context,
            "tool_used":tool_used
        }
    def validate_input(self, user_input):
        """校验用户输入是否符合规范"""
        input_data = {
            "question": user_input,
            "model": self.current_model,
            "temperature": self.current_temperature,
            "image": self.current_image 
        }
        
        # 特殊校验：如果选择了deepseek-vl2模型但没有上传图片
        if self.current_model == "Qwen/QVQ-72B-Preview" and not self.current_image:
            return False, "DeepSeek-VL2多模态模型需要上传图片"
            
        # 特殊校验：如果选择了非多模态模型但上传了图片
        if self.current_model != "Qwen/QVQ-72B-Preview" and self.current_image:
            return False, f"{self.current_model}模型不支持图像输入，请清除图片或切换模型"
            
        try:
            jsonschema.validate(instance=input_data, schema=INPUT_SCHEMA)
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            error_msg = f"输入格式错误: {e.message}"
            return False, error_msg
        except Exception as e:
            error_msg = f"输入校验失败: {str(e)}"
            return False, error_msg
    def validate_output(self, output_data):
        """校验输出数据是否符合规范"""
        try:
            jsonschema.validate(instance=output_data, schema=OUTPUT_SCHEMA)
            return True
        except jsonschema.exceptions.ValidationError as e:
            messagebox.showerror("输出错误", f"输出格式错误: {e.message}")
            return False
        except Exception as e:
            messagebox.showerror("输出错误", f"输出校验失败: {str(e)}")
            return False
    
    def send_message(self):
        """发送用户消息并获取AI回复"""
        
        user_input = self.input_box.get("1.0", tk.END).strip()
        print(f"send_message: 用户输入: {user_input}")       
        # 校验输入
        is_valid, error = self.validate_input(user_input)
        if not is_valid:
            messagebox.showerror("输入错误", error)
            return
            
        if self.is_waiting_response:
            messagebox.showinfo("提示", "请等待当前响应完成")
            return
            
        # 清空输入框并记录问题
        self.current_question = user_input
        self.input_box.delete("1.0", tk.END)
        
        # 更新UI
        self._append_message("user", user_input)
        self.chat_history.append({"role": "user", "content": user_input})
        self.memory_status.set(f"记忆状态: 保存了 {len(self.chat_history)//2} 轮对话")
        self.status_var.set(f"思考中... [{self.current_model}]")
        self.send_button.config(state=tk.DISABLED)
        self.is_waiting_response = True
        
        # 在新线程中调用API
        threading.Thread(target=self._process_user_question, daemon=True).start()
    def read_file(self, file_path:str):
        """读取文件内容"""
        if not self.file_enabled:
            print("read_file: 文件读取已禁用")
            return {"tool": "file", "result": "文件读取已禁用", "success": False}
            
        try:
            # 检查文件是否在允许的目录下
            allowed_dirs = self.tool_config.get("file", {}).get("allowed_dirs", [os.getcwd()])
            file_abs_path = os.path.abspath(file_path)
            is_allowed = False
            
            """for dir_path in allowed_dirs:
                dir_abs_path = os.path.abspath(dir_path)
                if file_abs_path.startswith(dir_abs_path):
                    is_allowed = True
                    print("read_file: 文件在允许的目录下")
                    break
                    
            if not is_allowed:
                print("read_file: 文件不在允许的目录中")
                return {"tool": "file", "result": "文件不在允许的目录中", "success": False}"""
                
            if not os.path.exists(file_path):
                print("read_file: 文件不存在")
                return {"tool": "file", "result": "文件不存在", "success": False}
                
            # 检查文件扩展名
            if file_path.lower().endswith('.docx'):
                # 处理 .docx 文件
                doc = Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])     
            elif file_path.lower().endswith('.pdf'):
                # 处理 .pdf
                doc = fitz.open(file_path)
                content = ""
                for page in doc:
                    content += page.get_text()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            # 限制内容长度，避免过大
            max_length = 3000
            truncated_content = content[:max_length] + ("..." if len(content) > max_length else "")
            print(f"read_file: 文件内容: {truncated_content}")
            return {"tool": "file", "result": f"文件内容: {truncated_content}", "success": True}
        except Exception as e:
            print(f"read_file: 文件读取失败: {str(e)}")
            return {"tool": "file", "result": f"文件读取失败: {str(e)}", "success": False}
    
    def get_weather(self, location: str):
        """获取指定地点的天气信息"""
        if not self.weather_enabled:
            print("get_weather: 天气查询已禁用")
            return {"tool": "weather", "result": "天气查询已禁用", "success": False}
            
        if not self.weather_api_key:
            print("get_weather: 未配置天气API密钥")
            return {"tool": "weather", "result": "未配置天气API密钥，请先在工具配置中设置", "success": False}
            
        try:
            # 使用OpenWeatherMap API获取天气信息
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.weather_api_key,
                "units": "metric"  # 摄氏温度
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                weather_desc = data['weather'][0]['description']
                temperature = data['main']['temp']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed']
                
                result = (f"当前{location}的天气状况：{weather_desc}。"
                         f"温度：{temperature}°C，湿度：{humidity}%，风速：{wind_speed}m/s。")
                
                print(f"get_weather: 天气查询成功: {result}")
                return {"tool": "weather", "result": result, "success": True}
            else:
                error_msg = f"天气查询失败，错误码：{response.status_code}，原因：{data.get('message', '未知错误')}"
                print(f"get_weather: {error_msg}")
                return {"tool": "weather", "result": error_msg, "success": False}
                
        except Exception as e:
            print(f"get_weather: 天气查询异常: {str(e)}")
            return {"tool": "weather", "result": f"天气查询异常: {str(e)}", "success": False}
    
    def web_search(self, query: str, num_results=10):
        """执行网络搜索"""
        if not self.search_enabled:
            print("web_search: 网络搜索已禁用")
            return {"tool": "search", "result": "网络搜索已禁用", "success": False}
            
        if not self.search_api_key:
            print("web_search: 未配置搜索API密钥")
            return {"tool": "search", "result": "未配置搜索API密钥，请先在工具配置中设置", "success": False}
            
        try:
            # 使用SerpAPI作为搜索后端
            #base_url = "https://serpapi.com/search"
            #params = {
            #    "q": query,
            #    "api_key": self.search_api_key,
            #    "engine": "google",
            #    "num": num_results
            #}            
            # 定义不同引擎的基础URL和参数
            search_engines = {
                "bing": {
                    "base_url": "https://serpapi.com/search",
                    "params": {"engine": "bing", "num": num_results}
                },
                "baidu": {
                    "base_url": "https://serpapi.com/search",
                    "params": {
                        "engine": "baidu",
                        "num": num_results,
                    }
                },
                "google": {  # 保留谷歌选项
                    "base_url": "https://serpapi.com/search",
                    "params": {"engine": "google", "num": num_results}
                }
            }
            print(f"web_search:当前选择的引擎: {self.search_provider}")
            engine_config = search_engines.get(self.search_provider, search_engines["google"])
            base_url = engine_config["base_url"]
            params = {
                "q": query,
                "api_key": self.search_api_key,
                **engine_config["params"]
            }
            #print(f"实际搜索URL: {base_url}?q={query}")  # 打印可直接访问的URL
            # 生成URL（正确方式）
            query_string = urlencode(params)
            full_url = f"{base_url}?{query_string}"

            print(f"web_search:实际搜索URL: {full_url}")            
            response = requests.get(full_url)
            data = response.json()
            print("完整API响应:")
            print(json.dumps(data, indent=2, ensure_ascii=False))  # 打印完整数据            
            if response.status_code == 200:
                if 'error' in data:
                    error_msg = f"搜索失败: {data['error']}"
                    print(f"web_search: {error_msg}")
                    return {"tool": "search", "result": error_msg, "success": False}
                    
                results = []
                if 'organic_results' in data:
                    for item in data['organic_results'][:num_results]:
                        title = item.get('title', '')
                        link = item.get('link', '')
                        snippet = item.get('snippet', '')
                        results.append(f"标题: {title}\n链接: {link}\n摘要: {snippet}\n\n")
                
                if results:
                    result_text = "搜索结果:\n\n" + "\n".join(results)
                    print(f"web_search: 搜索成功，返回{len(results)}条结果")
                    print(f"web_search: \n{result_text}")
                    return {"tool": "search", "result": result_text, "success": True}
                else:
                    print("web_search: 未找到搜索结果")
                    return {"tool": "search", "result": "未找到相关搜索结果", "success": False}
            else:
                error_msg = f"搜索请求失败，错误码：{response.status_code}"
                print(f"web_search: {error_msg}")
                return {"tool": "search", "result": error_msg, "success": False}
                
        except Exception as e:
            print(f"web_search: 搜索异常: {str(e)}")
            return {"tool": "search", "result": f"搜索异常: {str(e)}", "success": False}
    
    def should_call_file_tool(self, text):
        """判断是否需要调用文件读取工具"""
        if not self.file_enabled:
            print("should_call_file_tool: 文件读取已禁用")
            return False
            
        # 检查是否需要文件读取
        file_keywords = ["文件", "读取", "打开", "内容", "路径", "目录", ".txt", ".doc", ".pdf", ".docx"]
        if any(keyword in text for keyword in file_keywords):
            print("should_call_file_tool: 需要文件读取")
            match = re.search(r'([\w\s/\\().+?\-]+?\.(txt|docx|pdf|md|doc))', text, re.IGNORECASE)
            print(f"should_call_file_tool: 结果：{'success' if match is not None else 'failed'} match:", match)
            return match is not None
                
        return False
    
    def should_call_weather_tool(self, text):
        """判断是否需要调用天气查询工具"""
        if not self.weather_enabled:
            print("should_call_weather_tool: 天气查询已禁用")
            return False
            
        # 检查是否需要天气查询
        weather_keywords = ["天气", "温度", "气象", "预报", "晴", "雨", "风", "多云", "湿度", "气候", "weather", "temperature", "weather forecast", "sunny", "rain", "wind", "cloudy", "humidity"]
        location_pattern = location_pattern = r'([\u4e00-\u9fa5]{2,5}(市|县|区|镇|村|乡)?|[A-Za-z]+)\s*(天气|气候|weather|forecast)' 
        
        if any(keyword in text for keyword in weather_keywords):
            print("should_call_weather_tool: 需要天气查询")
            match = re.search(location_pattern, text)
            print(f"should_call_weather_tool: 结果：{'success' if match is not None else 'failed'} match:", match)
            return match is not None
                
        return False
    

    
    def should_call_search_tool(self, text):
        """判断是否需要调用网络搜索工具"""
        if not self.search_enabled:
            print("should_call_search_tool: 网络搜索已禁用")
            return False
            
        # 检查是否需要网络搜索
        search_keywords = ["搜索", "查询", "查找", "最新", "新闻", "资讯", "信息", "知识", "了解", "什么是", "是谁", "哪里", "何时", "为什么", "怎么样", "告诉我", "是啥"]
        
        # 如果问题中包含这些关键词，并且不是明确的文件或天气查询
        if any(keyword in text for keyword in search_keywords):
            #if not self.should_call_file_tool(text) and not self.should_call_weather_tool(text):
            print("should_call_search_tool: 需要网络搜索")
            return True
                
        # 如果问题是一个简单疑问句，也触发搜索
        question_patterns = [
            r'^[\u4e00-\u9fa5a-zA-Z0-9\s]*[?？]$',  # 以问号结尾
            r'^什么是[\u4e00-\u9fa5a-zA-Z0-9\s]+$',  # 以"什么是"开头
            r'^谁是[\u4e00-\u9fa5a-zA-Z0-9\s]+$',    # 以"谁是"开头
            r'^为什么[\u4e00-\u9fa5a-zA-Z0-9\s]+$',  # 以"为什么"开头
            r'^如何[\u4e00-\u9fa5a-zA-Z0-9\s]+$',    # 以"如何"开头
            r'^怎样[\u4e00-\u9fa5a-zA-Z0-9\s]+$',    # 以"怎样"开头
            r'^哪里[\u4e00-\u9fa5a-zA-Z0-9\s]+$',    # 以"哪里"开头
            r'^何时[\u4e00-\u9fa5a-zA-Z0-9\s]+$',    # 以"何时"开头
        ]
        
        for pattern in question_patterns:
            if re.match(pattern, text):
                print("should_call_search_tool: 简单疑问句，需要网络搜索")
                return True
                
        return False
    
    def _process_user_question(self):
        """处理用户问题，包括文件读取、天气查询和网络搜索工具调用"""
        try:
            # 判断是否需要调用工具
            tool_used = "none"
            tool_result = {"tool": "none", "result": "", "success": True}
            rag_context = ""
            # 优先级：搜索 > 天气 > 文件
            if self.should_call_search_tool(self.current_question):
                # 提取搜索关键词
                print("_process_user_question: 提取搜索关键词")
                # 移除可能的搜索指令前缀
                search_query = self.current_question
                prefixes = ["搜索", "查询", "查找", "告诉我", "了解", "什么是", "是谁", "哪里", "何时", "为什么", "怎么样"]
                for prefix in prefixes:
                    if search_query.startswith(prefix):
                        search_query = search_query[len(prefix):].strip()
                        break
                
                print(f"_process_user_question: 搜索关键词: {search_query}")
                tool_result = self.web_search(search_query)
                tool_used = "search"
                
                # 显示工具调用结果
                self.root.after(0, self._append_message, "tool", f"[网络搜索] {tool_result['result']}")
                
                # 如果工具调用失败，告知用户并继续使用LLM
                if not tool_result["success"]:
                    print("_process_user_question: 网络搜索失败，使用LLM继续处理问题")  
                    llm_response, rag_context = self._get_llm_response(f"网络搜索失败: {tool_result['result']}，请直接回答: {self.current_question}")
                else:
                    # 工具调用成功，使用工具结果生成最终回答
                    print("_process_user_question: 网络搜索成功，生成最终回答")
                    llm_response, rag_context = self._get_llm_response(f"根据网络搜索结果: {tool_result['result']}，回答用户问题: {self.current_question}")
                    
            elif self.should_call_weather_tool(self.current_question):
                # 提取地点
                print("_process_user_question: 提取地点")
                #location_pattern = r'([\u4e00-\u9fa5]{2,5}(市|县|区|镇|村|乡)?)'
                #location_pattern = r'([\u4e00-\u9fa5]{2,5}(市|县|区|镇|村|乡)?)\s*(天气|气候)'
                location_pattern = r'([\u4e00-\u9fa5]{2,8}(市|县|区|镇|村|乡)?|[A-Za-z]+)\s*(天气|气候|weather|forecast)'
                match = re.search(location_pattern, self.current_question)
                if match:
                    print(f"完整匹配: {match.group(0)}")   # 输出完整匹配内容
                    print(f"捕获组1: {match.group(1)}")  # 输出第一个捕获组
                location = match.group(1) if match else "北京市"  # 默认北京
                if location.startswith("查询"):
                    location = location[2:]
                if location.endswith("的"):
                    location = location[:-1]
                # 移除"今天"/"今日"前缀
                if location.startswith(("今天", "今日")):
                    location = location[2:]

                # 移除"今天"/"今日"后缀
                if location.endswith(("今天", "今日")):
                    location = location[:-2]  # 注意这里是-2，删除两个字符

                print(f"修改后 查询地点: {location}")
                if (not (location.isalpha() and location.isascii())) and "市" not in location: #中文城市缺少"市"
                    print(f"{location}缺少'市'")
                    location += "市"
                print(f"_process_user_question: 地点: {location}")
                tool_result = self.get_weather(location)
                tool_used = "weather"
                
                # 显示工具调用结果
                self.root.after(0, self._append_message, "tool", f"[天气查询] {tool_result['result']}")
                
                # 如果工具调用失败，告知用户并继续使用LLM
                if not tool_result["success"]:
                    print("_process_user_question: 天气查询失败，使用LLM继续处理问题")  
                    llm_response, rag_context = self._get_llm_response(f"天气查询失败: {tool_result['result']}，请直接回答: {self.current_question}")
                else:
                    # 工具调用成功，使用工具结果生成最终回答
                    print("_process_user_question: 天气查询成功，生成最终回答")
                    llm_response, rag_context = self._get_llm_response(f"根据当前天气信息: {tool_result['result']}，回答用户问题: {self.current_question}")
                    
            elif self.should_call_file_tool(self.current_question):
                # 提取文件路径
                print("_process_user_question: 提取文件路径")
                match = re.search(r'([\w\s/\\().+?:\-]+?\.(txt|docx|pdf|md|doc))', self.current_question, re.IGNORECASE)
                file_path = match.group(1) if match else ""
                
                if file_path:
                    idx = file_path.find(":")
                    if idx == -1 or idx == 0:
                        raise ValueError("文件路径格式错误")
                    file_path = file_path[idx-1:]
                    print(f"_process_user_question: 文件路径: {file_path}")
                    tool_result = self.read_file(file_path)
                    tool_used = "file"
                    
                    # 显示工具调用结果
                    self.root.after(0, self._append_message, "tool", f"[文件读取] {tool_result['result']}")
                    
                    # 如果工具调用失败，告知用户并继续使用LLM
                    if not tool_result["success"]:
                        print("_process_user_question: 文件读取失败，使用LLM继续处理问题")  
                        llm_response, rag_context = self._get_llm_response(f"文件读取失败: {tool_result['result']}，请直接回答: {self.current_question}")
                    else:
                        # 工具调用成功，使用工具结果生成最终回答
                        print("_process_user_question: 文件读取成功，生成最终回答")
                        llm_response, rag_context = self._get_llm_response(f"根据文件内容: {tool_result['result']}，回答用户问题: {self.current_question}")
                else:
                    print("_process_user_question: 文件读取失败，使用原始问题生成最终回答")
                    llm_response, rag_context = self._get_llm_response(self.current_question)
            else:
                # 不需要调用工具，直接使用LLM
                print("_process_user_question: 不需要调用工具,直接使用LLM")
                llm_response, rag_context = self._get_llm_response(self.current_question)
                
            # 格式化输出
            output_data = self.format_output(llm_response, rag_context = rag_context,tool_used=tool_used,)
            
            # 校验输出
            if not self.validate_output(output_data):
                print("_process_user_question: 输出格式校验失败")
                self.root.after(0, self._append_message, "error", "输出格式校验失败")
                return
                
            # 更新聊天历史
            self.chat_history.append({
                "role": "assistant", 
                "content": output_data["content"],
                "model": output_data["model"],
                "timestamp": output_data["timestamp"],
                "rag_context": rag_context,
                "tool_used": output_data["tool_used"]
            })
            # 限制历史记录长度
            if len(self.chat_history) > self.max_history_length * 2:
                # 保留最近的对话
                self.chat_history = self.chat_history[-self.max_history_length*2:]
                
            self.root.after(0, lambda: self.memory_status.set(
                f"记忆状态: 保存了 {len(self.chat_history)//2} 轮对话"))
            
        except Exception as e:
            print(f"_process_user_question: 处理用户问题时发生错误: {str(e)}")
            error_msg = f"处理用户问题时发生错误: {str(e)}"
            error_output = self.format_output(error_msg, status="error", tool_used="none")
            self.validate_output(error_output)
            self.root.after(0, self._append_message, "error", error_msg)
            
        finally:
            # 恢复UI状态
            self.root.after(0, self._update_ui_after_response)
    def _get_llm_response(self, prompt):
        """获取LLM响应"""
        try:
            messages = self._prepare_chat_history()
            # 如果启用了RAG且有知识库，检索相关上下文
            rag_context = ""
            if self.rag_enabled.get() and self.knowledge_base:
                rag_context = self.retrieve_context(self.current_question, top_k=self.rag_context_var.get())
                
                # 更新RAG上下文显示
                self.root.after(0, self._update_rag_display, rag_context)
                # 将上下文添加到消息中
                if rag_context:
                    messages.insert(0, {
                        "role": "system",
                        "content": f"根据以下上下文回答问题:\n{rag_context}\n\n如果上下文没有相关信息，请根据你的知识回答。"
                    })
            
            # 特殊处理多模态模型
            if self.current_model == "Qwen/QVQ-72B-Preview" and self.current_image:
                # 构造多模态消息
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.current_image}"
                            }
                        }
                    ]
                })
            else:
                # 普通文本消息
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            # 创建API请求
            print("_get_llm_response触发:正在获取LLM响应...")
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                stream=True,  # 启用流式响应
                temperature=self.current_temperature
            )

            # 开始流式显示响应
            self.root.after(0, self._append_message, "assistant", "")            
            full_response = ""
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    if content:
                        full_response += content
                        # 实时更新UI
                        self.root.after(0, self._append_message, "assistant", content, True)
            print(f"_get_llm_response: 响应结果full_response: {full_response}")
            return full_response, rag_context
        except Exception as e:
            print(f"_get_llm_response: 获 {str(e)}")
            return f"获取AI响应失败: {str(e)}" , ""       
        
    def _get_ai_response(self):
        """获取AI响应并流式显示"""
        try:
            # 创建API请求 - 使用完整的对话历史
            messages = self._prepare_chat_history()
            
            # 如果启用了RAG且有知识库，检索相关上下文
            rag_context = ""
            if self.rag_enabled.get() and self.knowledge_base:
                rag_context = self.retrieve_context(self.current_question, top_k=self.rag_context_var.get())
                
                # 更新RAG上下文显示
                self.root.after(0, self._update_rag_display, rag_context)
                # 将上下文添加到消息中
                if rag_context:
                    messages.insert(0, {
                        "role": "system",
                        "content": f"根据以下上下文回答问题:\n{rag_context}\n\n如果上下文没有相关信息，请根据你的知识回答。"
                    })
            
            # 特殊处理多模态模型
            if self.current_model == "Qwen/QVQ-72B-Preview" and self.current_image:
                # 构造多模态消息
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.current_question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.current_image}"
                            }
                        }
                    ]
                })
            else:
                # 普通文本消息
                messages.append({
                    "role": "user",
                    "content": self.current_question
                })
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                stream=True,
                temperature=self.current_temperature
            )
            
            # 开始流式显示响应
            self.root.after(0, self._append_message, "assistant", "")
            full_response = ""
            
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        full_response += content
                        self.root.after(0, self._append_message, "assistant", content, True)
            
            # 格式化最终输出并校验
            output_data = self.format_output(full_response, rag_context=rag_context)
            if not self.validate_output(output_data):
                self.root.after(0, self._append_message, "error", "输出格式校验失败")
                return
                
            # 更新聊天历史
            self.chat_history.append({
                "role": "assistant", 
                "content": full_response,
                "model": self.current_model,
                "timestamp": output_data["timestamp"],
                "rag_context": rag_context,
                "tool_used": output_data.get("tool_used", "")
            })
            
            # 限制历史记录长度
            if len(self.chat_history) > self.max_history_length * 2:
                # 保留最近的对话
                self.chat_history = self.chat_history[-self.max_history_length*2:]
                
            self.root.after(0, lambda: self.memory_status.set(
                f"记忆状态: 保存了 {len(self.chat_history)//2} 轮对话"))
            
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            error_output = self.format_output(error_msg, status="error")
            self.validate_output(error_output)  # 校验错误输出
            self.root.after(0, self._append_message, "error", error_msg)
            
        finally:
            # 恢复UI状态
            self.root.after(0, self._update_ui_after_response)
            
    def _prepare_chat_history(self):
        """准备用于API调用的对话历史"""
        # 如果历史记录太长，只保留最近的几轮对话
        try:
            max_history = min(len(self.chat_history), self.max_history_length * 2)
            messages = self.chat_history[-max_history:]
            # 确保角色名称符合API要求
            for msg in messages:
                if msg["role"] == "ai":
                    msg["role"] = "assistant"
        except Exception as e:
            print(f"_prepare_chat_history: 错误: {str(e)}")
            return []

        return messages
        
    def _append_message(self, role, message, append=False):
        """向聊天窗口添加消息，并使用rich输出Markdown格式"""
        self.chat_display.config(state=tk.NORMAL)
        
        if not append:
            # 添加新消息
            if role == "user":
                self.chat_display.insert(tk.END, "\n\n你:\n", "user_header")
                self.chat_display.insert(tk.END, message + "\n\n", "user_message")
                # 使用rich输出Markdown格式
                console.print(Markdown(f"**你**:\n{message}"))
            elif role == "assistant":
                self.chat_display.insert(tk.END, f"AI ({self.current_model}):\n", "ai_header")
                self.chat_display.insert(tk.END, message, "ai_message")
                # 使用rich输出Markdown格式
                console.print(Markdown(f"**AI ({self.current_model})**:\n{message}"))
            elif role == "tool":
                self.chat_display.insert(tk.END, f"[工具] {message}\n\n", "tool_message")
            elif role == "error":
                self.chat_display.insert(tk.END, message + "\n\n", "error")
                console.print(f"[bold red]错误:[/bold red] {message}")
        else:
            # 追加到现有消息
            self.chat_display.insert(tk.END, message, "ai_message")
            #self.chat_display.insert(tk.END, message)
            
        # 滚动到底部
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def _update_rag_display(self, context):
        """更新RAG上下文显示，并使用rich输出Markdown格式"""
        self.rag_context_display.config(state=tk.NORMAL)
        self.rag_context_display.delete(1.0, tk.END)
        
        if context:
            self.rag_context_display.insert(tk.END, "检索到的相关上下文:\n\n", "rag_header")
            self.rag_context_display.insert(tk.END, context, "rag_content")
            # 使用rich输出Markdown格式
            console.print(Markdown("**检索到的相关上下文**:\n" + context))
        else:
            self.rag_context_display.insert(tk.END, "未检索到相关上下文\n", "rag_header")
            console.print("[italic]未检索到相关上下文[/italic]")
        
        self.rag_context_display.config(state=tk.DISABLED)
        self.rag_context_display.tag_config("rag_header", foreground="#5D4037", font=(self.font_family[0], 9, "bold"))
        self.rag_context_display.tag_config("rag_content", foreground="#5D4037", font=(self.font_family[0], 9))
    
    def _update_ui_after_response(self):
        """响应完成后更新UI状态"""
        self.status_var.set("准备就绪")
        self.send_button.config(state=tk.NORMAL)
        self.is_waiting_response = False
        
    def clear_memory(self):
        """清除对话记忆"""
        if messagebox.askyesno("确认", "确定要清除所有对话记忆吗？"):
            self.chat_history = []
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.memory_status.set("记忆状态: 记忆已清除")
            self.status_var.set("记忆已清除")
            
    def save_memory(self):
        """保存对话记忆到文件，文件名基于对话内容"""
        try:
            # 生成基于对话内容的文件名
            if self.chat_history:
                # 获取第一条用户消息作为标题
                first_user_message = ""
                for msg in self.chat_history:
                    if msg["role"] == "user":
                        first_user_message = msg["content"]
                        break
                
                # 清理消息内容用于文件名
                if first_user_message:
                    # 删除特殊字符和多余空格
                    clean_title = re.sub(r'[\\/*?:"<>|]', '', first_user_message)
                    clean_title = clean_title.replace('\n', ' ').replace('\r', '')
                    clean_title = clean_title.strip()
                    
                    # 截取前15个字符作为标题
                    if len(clean_title) > 15:
                        clean_title = clean_title[:15] + "..."
                else:
                    clean_title = "无标题"
            else:
                clean_title = "空对话"
            
            # 添加时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{clean_title}_{timestamp}.json"
            
            # 确保文件名长度合理
            if len(filename) > 100:
                filename = f"chat_{timestamp}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                
            self.status_var.set(f"记忆已保存到 {filename}")
            self.memory_status.set(f"记忆状态: 已保存 {len(self.chat_history)//2} 轮对话到 {filename}")
            
            # 刷新文件列表
            self.refresh_file_list()
            return filename
        except Exception as e:
            messagebox.showerror("保存错误", f"保存记忆失败: {str(e)}")
            return None
            
    def load_memory(self, filename):
        """从指定文件加载对话记忆"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.chat_history = json.load(f)
                
            # 更新UI显示
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            
            # 重新显示所有历史消息
            for msg in self.chat_history:
                if msg["role"] == "user":
                    self._append_message("user", msg["content"])
                elif msg["role"] == "assistant":
                    self._append_message("assistant", msg["content"])
            
            self.chat_display.config(state=tk.DISABLED)
            self.memory_status.set(f"记忆状态: 已加载 {len(self.chat_history)//2} 轮对话")
            self.status_var.set(f"已从文件加载记忆: {os.path.basename(filename)}")
            
            # 刷新文件列表以更新修改时间
            self.refresh_file_list()
            return True
        except FileNotFoundError:
            messagebox.showerror("加载错误", f"未找到文件: {filename}")
            return False
        except Exception as e:
            messagebox.showerror("加载错误", f"加载记忆失败: {str(e)}")
            return False
            
    def load_memory_dialog(self):
        """打开文件对话框加载记忆"""
        filetypes = [("JSON文件", "*.json"), ("所有文件", "*.*")]
        filename = filedialog.askopenfilename(
            title="选择记忆文件",
            initialdir=".",
            filetypes=filetypes
        )
        
        if filename:
            self.load_memory(filename)
            
    def view_saved_memories(self):
        """查看已保存的记忆文件"""
        # 获取当前目录下所有记忆文件
        memory_files = [f for f in os.listdir(".") if f.startswith("chat_") and f.endswith(".json")]
        
        if not memory_files:
            messagebox.showinfo("保存的记忆", "没有找到保存的记忆文件")
            return
            
        # 创建文件选择对话框
        file_dialog = tk.Toplevel(self.root)
        file_dialog.title("选择要加载的记忆")
        file_dialog.geometry("500x400")
        file_dialog.resizable(True, True)
        file_dialog.transient(self.root)
        file_dialog.grab_set()
        
        # 文件列表框架
        frame = ttk.Frame(file_dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文件列表
        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, font=(self.font_family[0], 10))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.config(yscrollcommand=scrollbar.set)
        
        # 添加文件到列表
        for file in sorted(memory_files, reverse=True):  # 最新的文件在前
            listbox.insert(tk.END, file)
        
        # 布局
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮框架
        btn_frame = ttk.Frame(file_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 加载按钮
        load_btn = ttk.Button(
            btn_frame, text="加载选中的记忆",
            command=lambda: self._load_selected_memory(listbox, file_dialog),
            style="Memory.TButton"
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 删除按钮
        delete_btn = ttk.Button(
            btn_frame, text="删除选中的文件",
            command=lambda: self._delete_selected_memory(listbox),
            style="Accent.TButton"
        )
        delete_btn.pack(side=tk.RIGHT, padx=5)
        
        # 取消按钮
        cancel_btn = ttk.Button(
            btn_frame, text="取消",
            command=file_dialog.destroy
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
    def _load_selected_memory(self, listbox, dialog):
        """加载选中的记忆文件"""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("选择错误", "请选择一个记忆文件")
            return
            
        filename = listbox.get(selection[0])
        self.load_memory(filename)
        dialog.destroy()
        
    def _delete_selected_memory(self, listbox):
        """删除选中的记忆文件"""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("选择错误", "请选择一个记忆文件")
            return
            
        filename = listbox.get(selection[0])
        if messagebox.askyesno("确认删除", f"确定要永久删除文件 {filename} 吗？"):
            try:
                os.remove(filename)
                listbox.delete(selection[0])
                messagebox.showinfo("成功", f"文件 {filename} 已删除")
            except Exception as e:
                messagebox.showerror("删除错误", f"删除文件失败: {str(e)}")
    
    # ================= RAG 功能实现 =================
    
    def load_knowledge_dialog(self):
        """打开文件对话框加载知识文档"""
        filetypes = [
            ("文本文件", "*.txt"),
            ("Markdown文件", "*.md"),
            ("PDF文件", "*.pdf"),
            ("所有文件", "*.*")
        ]
        
        filenames = filedialog.askopenfilenames(
            title="选择知识文档",
            initialdir=".",
            filetypes=filetypes
        )
        
        if filenames:
            for filename in filenames:
                self.load_knowledge_file(filename)
    
    def load_knowledge_file(self, filename):
        """加载知识文档到知识库"""
        try:
            # 检测文件编码
            with open(filename, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # 读取文件内容
            if filename.lower().endswith('.pdf'):
                # PDF文件处理
                try:
                    with open(filename, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                except Exception as e:
                    messagebox.showerror("PDF读取错误", f"读取PDF文件失败: {str(e)}")
                    return False
            else:
                # 文本文件处理
                with open(filename, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
            
            # 分块处理文档
            chunks = self.chunk_text(text)
            
            # 添加到知识库
            for chunk in chunks:
                self.knowledge_base.append({
                    "source": os.path.basename(filename),
                    "content": chunk
                })
            
            # 更新向量索引
            self.update_vector_index()
            
            # 更新状态
            doc_count = len(set(doc['source'] for doc in self.knowledge_base))
            self.knowledge_status.set(f"知识库: {len(self.knowledge_base)} 个片段 ({doc_count} 个文档)")
            self.status_var.set(f"已加载知识文档: {os.path.basename(filename)}")
            return True
        except Exception as e:
            messagebox.showerror("加载错误", f"加载知识文档失败: {str(e)}")
            return False
    
    def chunk_text(self, text, chunk_size=500):
        """将文本分割成较小的块，使用中文分词"""
        # 使用jieba进行分词
        words = list(jieba.cut(text))
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # 如果当前块长度加上新词长度超过阈值，并且当前块不为空
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += word_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def update_vector_index(self):
        """更新向量索引，使用中文分词"""
        if not self.knowledge_base:
            self.tfidf_matrix = None
            return
        
        # 提取所有知识片段内容
        documents = [doc["content"] for doc in self.knowledge_base]
        
        # 使用自定义的中文分词器
        self.vectorizer = TfidfVectorizer(tokenizer=self.chinese_tokenizer)
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def retrieve_context(self, query, top_k=3):
        """检索与查询最相关的上下文，使用中文分词"""
        if not self.knowledge_base or self.tfidf_matrix is None:
            return ""
        
        # 对查询进行分词处理
        tokenized_query = " ".join(self.chinese_tokenizer(query))
        
        # 向量化查询
        query_vec = self.vectorizer.transform([tokenized_query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 将片段按相似度降序排序，并记录索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 初始化
        selected_contexts = []
        selected_sources = set()
        
        # 遍历排序后的索引
        for idx in sorted_indices:
            # 如果已经选够了，则退出
            if len(selected_contexts) >= top_k:
                break
                
            doc = self.knowledge_base[idx]
            
            # 确保每个文档只选一个片段
                
            # 否则，选取该片段
            selected_contexts.append(doc)
            selected_sources.add(doc['source'])
        
        # 构建上下文
        context = ""
        for doc in selected_contexts:
            context += f"来源: {doc['source']}\n内容: {doc['content']}\n\n"
        
        return context.strip()
    
    def clear_knowledge_base(self):
        """清空知识库"""
        if messagebox.askyesno("确认", "确定要清空知识库吗？"):
            self.knowledge_base = []
            self.tfidf_matrix = None
            self.knowledge_status.set("知识库: 0 个文档")
            self.status_var.set("知识库已清空")
            self._update_rag_display("")
    
    def view_knowledge_base(self):
        """查看知识库内容"""
        if not self.knowledge_base:
            messagebox.showinfo("知识库", "知识库为空")
            return
        
        # 创建知识库查看窗口
        knowledge_dialog = tk.Toplevel(self.root)
        knowledge_dialog.title("知识库内容")
        knowledge_dialog.geometry("700x500")
        knowledge_dialog.resizable(True, True)
        knowledge_dialog.transient(self.root)
        knowledge_dialog.grab_set()
        
        # 主框架
        main_frame = ttk.Frame(knowledge_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 文档列表框架
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文档列表
        doc_list = tk.Listbox(list_frame, selectmode=tk.SINGLE, font=(self.font_family[0], 10))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=doc_list.yview)
        doc_list.config(yscrollcommand=scrollbar.set)
        
        # 添加文档到列表（按来源分组）
        doc_sources = {}
        for doc in self.knowledge_base:
            if doc["source"] not in doc_sources:
                doc_sources[doc["source"]] = []
            doc_sources[doc["source"]].append(doc["content"])
        
        for source, chunks in doc_sources.items():
            doc_list.insert(tk.END, f"文档: {source} ({len(chunks)} 个片段)")
        
        # 布局
        doc_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 内容显示框架
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        ttk.Label(
            content_frame, text="文档内容:",
            font=(self.font_family[0], 10, "bold")
        ).pack(anchor=tk.W)
        
        content_display = scrolledtext.ScrolledText(
            content_frame, wrap=tk.WORD, height=10,
            bg="#F8F8F8", fg="#333333",
            font=(self.font_family[0], 9)
        )
        content_display.pack(fill=tk.BOTH, expand=True)
        content_display.config(state=tk.DISABLED)
        
        # 绑定选择事件
        def on_doc_select(event):
            selection = doc_list.curselection()
            if not selection:
                return
                
            source = doc_list.get(selection[0]).split(": ")[1].split(" (")[0]
            content_display.config(state=tk.NORMAL)
            content_display.delete(1.0, tk.END)
            
            # 显示该文档的所有片段
            for doc in self.knowledge_base:
                if doc["source"] == source:
                    content_display.insert(tk.END, f"片段 {self.knowledge_base.index(doc)+1}:\n", "fragment_header")
                    content_display.insert(tk.END, doc["content"] + "\n\n", "fragment_content")
            
            content_display.config(state=tk.DISABLED)
            content_display.tag_config("fragment_header", foreground="#5D4037", font=(self.font_family[0], 9, "bold"))
            content_display.tag_config("fragment_content", foreground="#5D4037", font=(self.font_family[0], 9))
        
        doc_list.bind("<<ListboxSelect>>", on_doc_select)
        
        # 关闭按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        close_btn = ttk.Button(
            btn_frame, text="关闭",
            command=knowledge_dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT)
    def run_file_config_gui(self):
        """打开文件读取配置GUI"""
        config_window = tk.Toplevel(self.root)
        config_window.title("文件读取配置")
        config_window.geometry("500x300")
        config_window.resizable(False, False)
        
        # 允许的文件目录
        ttk.Label(config_window, text="允许的文件目录 (用逗号分隔):").pack(padx=10, pady=5, anchor="w")
        self.dirs_entry = ttk.Entry(config_window, width=40)
        allowed_dirs = ", ".join(self.tool_config.get("file", {}).get("allowed_dirs", [os.getcwd()]))
        self.dirs_entry.insert(0, allowed_dirs)
        self.dirs_entry.pack(padx=10, pady=5, fill=tk.X)
        
        def save_config():
            """保存配置"""
            print("save_config: 保存配置")
            dirs = [d.strip() for d in self.dirs_entry.get().split(",") if d.strip()]
            self.tool_config["file"]["allowed_dirs"] = dirs
            
            with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.tool_config, f)
                
            messagebox.showinfo("成功", "文件读取配置已保存")
            config_window.destroy()
        
        ttk.Button(config_window, text="保存配置", command=save_config).pack(padx=10, pady=10)

    def run_weather_config_gui(self):
        """打开天气查询配置GUI"""
        config_window = tk.Toplevel(self.root)
        config_window.title("天气查询配置")
        config_window.geometry("500x200")
        config_window.resizable(False, False)
        
        # API密钥
        ttk.Label(config_window, text="OpenWeatherMap API密钥:").pack(padx=10, pady=5, anchor="w")
        self.api_key_entry = ttk.Entry(config_window, width=40, show="*")
        self.api_key_entry.insert(0, self.weather_api_key)
        self.api_key_entry.pack(padx=10, pady=5, fill=tk.X)
        
        # 提示信息
        ttk.Label(
            config_window, 
            text="获取API密钥: 访问 https://openweathermap.org/api 注册账号并获取API密钥",
            foreground="#666666",
            font=(self.font_family[0], 9)
        ).pack(padx=10, pady=5, anchor="w")
        
        def save_config():
            """保存配置"""
            print("save_config: 保存天气配置")
            api_key = self.api_key_entry.get().strip()
            self.tool_config["weather"]["api_key"] = api_key
            self.weather_api_key = api_key
            
            with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.tool_config, f)
                
            messagebox.showinfo("成功", "天气查询配置已保存")
            config_window.destroy()
        
        ttk.Button(config_window, text="保存配置", command=save_config).pack(padx=10, pady=10)

    def run_search_config_gui(self):
        """打开网络搜索配置GUI"""
        config_window = tk.Toplevel(self.root)
        config_window.title("网络搜索配置")
        config_window.geometry("500x300")
        config_window.resizable(False, False)
        
        # API密钥
        ttk.Label(config_window, text="SerpAPI API密钥:").pack(padx=10, pady=5, anchor="w")
        self.search_api_key_entry = ttk.Entry(config_window, width=40, show="*")
        self.search_api_key_entry.insert(0, self.search_api_key)
        self.search_api_key_entry.pack(padx=10, pady=5, fill=tk.X)
        
        # 搜索提供商
        ttk.Label(config_window, text="搜索提供商:").pack(padx=10, pady=5, anchor="w")
        self.search_provider_combo = ttk.Combobox(
            config_window,
            values=["google", "bing", "baidu"],
            state="readonly",
            width=20
        )
        self.search_provider_combo.set(self.search_provider)
        self.search_provider_combo.pack(padx=10, pady=5, fill=tk.X)
        
        # 提示信息
        ttk.Label(
            config_window, 
            text="获取API密钥: 访问 https://serpapi.com/ 注册账号并获取API密钥\n"
                "支持搜索引擎: Google、Bing、百度",
            foreground="#666666",
            font=(self.font_family[0], 9)
        ).pack(padx=10, pady=5, anchor="w")        
        def save_config():
            """保存配置"""
            print("save_config: 保存搜索配置")
            api_key = self.search_api_key_entry.get().strip()
            provider = self.search_provider_combo.get()
            
            self.tool_config["search"]["api_key"] = api_key
            self.tool_config["search"]["provider"] = provider
            self.search_api_key = api_key
            self.search_provider = provider
            
            with open(TOOL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.tool_config, f)
                
            messagebox.showinfo("成功", "网络搜索配置已保存")
            config_window.destroy()
        
        ttk.Button(config_window, text="保存配置", command=save_config).pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()