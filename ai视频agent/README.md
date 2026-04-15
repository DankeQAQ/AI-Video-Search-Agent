# ai-ip-agent
这是一个基于 DeepSeek、TMDB 和 Tavily API 构建的智能视频搜索助手。它能理解用户的自然语言需求，自动补全影视元数据，并在全球范围内寻找最优质、无 IP 限制的播放资源。输入自然语言需求，系统自动完成意图解析、元数据补全、检索重排与结果展示，尽可能返回可用度更高的播放入口。

## 简介

本项目面向“模糊找片”场景，核心目标是提升搜索结果的相关性与可访问性。  
通过多源搜索、标题语义过滤、域名分级与前端风险提示，帮助用户更快定位可观看资源。

## 功能亮点

- **全自动流水线（核心卖点）**
  - 一次点击自动执行：意图解析 -> TMDB 对齐 -> 多轮检索 -> 过滤重排 -> 前端展示。
  - 支持手动分步调试（解析 / TMDB / 搜索）以便定位问题。
- **海外资源优化（核心卖点）**
  - 检索策略优先覆盖海外常用可访问资源域名。
  - 支持海外模式提示，对可能受地域限制链接给出访问说明。
  - 在排序与标签层面强化 YouTube / B 站等稳定来源。
- **搜索质量增强**
  - 多查询并发（主查询 + 平台组合查询 + 定向补充查询）。
  - 标题语义匹配与成人敏感词过滤，减少垃圾页面与误命中。
  - 结果分层展示：优先结果 + “更多结果”折叠区。
- **可解释结果展示**
  - 每条结果展示域名来源标签（官方/推荐、第三方资源）。
  - 对第三方资源追加隐私与弹窗风险提示。

## 技术栈

- **后端**：FastAPI、Pydantic、HTTPX、Uvicorn
- **AI / LLM**：LangChain、OpenAI 兼容接口
- **搜索**：Tavily、Serper
- **元数据**：TMDB API
- **前端**：原生 HTML / CSS / JavaScript（静态页面）

## 安装步骤

1. 克隆项目并进入目录
   - `git clone <your-repo-url>`
   - `cd ai视频agent`
2. 创建虚拟环境
   - `python3 -m venv .venv`
3. 激活虚拟环境
   - macOS/Linux: `source .venv/bin/activate`
4. 安装依赖
   - `pip install -r requirements.txt`
5. 启动服务
   - `uvicorn app.main:app --reload`
6. 打开浏览器访问首页（默认 `http://127.0.0.1:8000`）

## 环境变量配置说明

请在项目根目录创建 `.env`，至少配置以下变量（按实际服务商填写）：

- `TAVILY_API_KEY`：Tavily 搜索 Key
- `SERPER_API_KEY`：Serper 搜索 Key（可选）
- `TMDB_API_KEY`：TMDB API Key 或 Read Access Token
- `OPENAI_API_KEY`：LLM Key
- `OPENAI_BASE_URL`：OpenAI 兼容网关地址（可选）
- `OPENAI_MODEL`：模型名称（如 `gpt-4o-mini` / 兼容模型）
- `SEARCH_PROVIDER`：`tavily` 或 `serper`
- `SEARCH_LLM_CURATE`：是否启用 LLM 精选（如 `true/false`）

> 提示：若未正确配置密钥，相关流程会在接口层返回可读错误信息。

## 免责声明

本项目仅用于**学习交流与技术演示**，不提供任何视频内容存储或分发服务。
