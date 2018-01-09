---
title: "Habits"
author: John Doe
date: March 22, 2005
output: revealjs::revealjs_presentation
---

<style type="text/css">
    p { text-align: left; }
</style>

<h1> Markdown Slides </h1>

# 1. 概述

## 使用vscode 

- 装vscode编辑器
- 在vscode插件，revealjs
- 编辑文档
- 然后在vscode中，按F1，进入命令行
    - 选：Revealjs:show presentation in slide

这种方式，需要自己手工加第一页slide的分隔符“---”。

另外一个工具可以自动添加分隔符

## 使用RStudio

![using RStudio and revealjs to render your markdown to get the slide ](./DR/RStudio.jpg)


# 2. 使用Markdown编写Slide

## 2.1. 基本流程

- 建议阅读[revealjs官方文档]<https://github.com/hakimel/reveal.js>

- 重点：
    - 默认veticalSeparator是“--”，Separator是“---”。
    - 在Markdown中通过添加Separator实现分页, 前后加空行。
    - 在一级标题前加Separator, 一级标题内的用veticalSeparator。
        - 整体效果主是第一行Slide都是一级标题，标题内的都在页向下的历史



## 2.2. 个人定制

### 2.2.1. 文本段落左对齐

在文档前面加上如下内容

```

<style type="text/css">
    p { text-align: left; }
</style>

```

基本的文本格式如下：

```

--- //revealjs参数设置，必须在第一行
theme : "white"
transition: "convex"
separator: "\n---\n"
veticalSeparator: "\n--\n"
---

<style type="text/css">   //加入个性化的css参数
    p { text-align: left; }
</style>

The rest are the markdown main content. //剩下是md文本内容

```

### 2.2.2. 自动生成separator

# 3. 样本

```markdown

^# 1 This is h1
--
^### 1.1
--
^### 1.2 
---
^# 2 This is h1

```
# 4. 其他实现方式

- Install local pandoc
    - **slidy** style works fine to me
        - pandoc 0x15_MAL_免杀原理与实践.md -o slides.html -t slidy -s
- pandoc plugin for vscode
- 
- revealjs plugin for vscode also works

# 5. 参考文献

This is the <https://linux.cn/article-4080-2.html> introduce how to produce slides with markdown.


