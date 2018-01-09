---
title: "Markdown slide"
author: Wildlinux
date: Jan 9, 2018
output: revealjs::revealjs_presentation
---

<style type="text/css">
    p { text-align: left; }
</style>

<h1> Markdown Slides </h1>

# 1. 概述

## 效果

- 可以使用纯文本（markdown）实现以下效果哦。
- 其意义在于内容与显示的分离。
- 以下内容就是本文档转在SLIDE的效果。

![It' cool ](./DR/Md_slides.gif)

## 实现方法1：使用vscode 

- 装vscode编辑器
- 在vscode安装插件，revealjs
- 编辑文档你的slide文档
- 然后在vscode中，按F1，进入命令行
    - 选：Revealjs:show presentation in slide

- 缺点：
  - 这种方式，需要自己手工加每一页slide的分隔符“---”。
  - 我还没找到现成的解决方案，但肯定有。

另外一个工具可以自动添加分隔符

## 实现方法2：使用RStudio

![using RStudio and revealjs to render your markdown to get the slide ](./DR/RStudio.jpg)

- 优点：
 - 自动为一二级标题切分slide页。
 - 有些需要额外分布的，需要手工加“---”或“--”。
 - 这样在写作时就维护一份文档就可以了。
 
- 缺点：
  - RStudio非常强**大**，用来做这个有点笨重
  - 没有直接预览。
  
# 2. 使用Markdown编写Slide

## 2.1. 基本流程

- Markdown本身的语法自行学习吧。
- revealjs
  - 建议阅读[revealjs官方文档]<https://github.com/hakimel/reveal.js>
- 重点：
    - 默认veticalSeparator是“--”，Separator是“---”。
    - 在Markdown中通过添加Separator实现分页, 前后加空行。
    - 在一级标题前加Separator, 一级标题内的用veticalSeparator。
    - 整体效果主是
      - 左右键会在一级标题级切换；上下键可在一级标题内切换。
      - 第一行Slide都是一级标题，标题内的本级标题下的页面。



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

- 使用RStudio实现。

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
        - pandoc yourfile.md -o slides.html -t slidy -s
- pandoc plugin for vscode
- revealjs plugin for vscode also works

This is the <https://linux.cn/article-4080-2.html> introduce how to produce slides with markdown.


