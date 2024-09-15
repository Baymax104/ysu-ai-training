## 目录结构

- `deploy`: 完成模型部署
  - `cli.py`: 命令行部署脚本
  - `web_gradio.py`: Gradio部署脚本
- `finetune`: 完成Chat3-6B微调
  - `configs`: 微调配置文件，仅使用LORA微调
  - `src`: 微调训练源代码
  - `finetune.sh`: 训练程序启动脚本
  - `inference.sh`: 模型推理启动脚本
- `logs`: 包含微调前后效果截图
  - `base`: 微调前
  - `finetuning`: 微调后
- `preprocessing`: 完成数据预处理
  - `data`: 预处理后的数据
  - `preprocessing.ipynb`: 预处理代码
- `spider`: 完成数据爬取
  - `src`: 爬虫源代码