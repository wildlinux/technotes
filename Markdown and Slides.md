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

## 1.1 效果

- 可以使用纯文本（markdown）实现以下效果哦。
- 以下内容就是当前网页转为SLIDE的效果：一份文档，多种用途。

- 本文中说到的slide，起到PPT一样的展示效果，但是一个动态网页，所以是在浏览器中打开的。
- 优点很多哦，完全跨平台，不用装任何软件就能直接展示，有浏览器就行。

![It' cool ](./DR/Md_slides.gif)

- 其意义在于维护简单了
  - 当前这个网页即是一份详细的操作记录。
  - 也可以稍加处理就成了一份可以展示、讲解的PPT。

> 可以通过简单的处理过滤掉文档中不想做成SLIDE的内容部分，又基本不改变文档内容。 

## 1.2 实现方法1：使用vscode 

- 安装vscode编辑器
- 在vscode安装插件，revealjs
- 正常编辑你的markdown文档
- 然后在vscode中，按F1，进入命令行
    - 选：Revealjs:show presentation in slide
    - Vscode会自动把md转换为slide模式。
- 优点：
  - vscode针对markdown插件非常之多，非常好用。
  - 可以直接预览效果。
- 缺点：
  - 这种方式，需要自己手工加每一页slide的分隔符“---”。
  - 我还没找到现成的解决方案，但肯定有。后期有空了再找吧。

下图是使用vscode的效果。动图开始是去年写的原始的文档网页版，git在oschina上的。动图后面左侧是文档原文，右侧是展示效果。同一个文档，就可以做展示了，不用费心做PPT了。当然在实际写文档时要考虑到展示的需求，适当合理组织内容。

![It' cool ](./DR/vscode.gif)

又不想手工分页，就找了下面这个工具，可以自动添加分隔符。

## 1.3 实现方法2：使用RStudio

按图中0-4步做就可以了。就是RStudio有点大。功能也是非常之强悍。可以写数学公式，直接显示运行结果图到Slide上。

![using RStudio and revealjs to render your markdown to get the slide ](./DR/RStudio.jpg)

- 优点：
  - 自动为一二级标题切分slide页。
  - 有些需要额外分页的，需要手工加“--”（一级标题下的内容部分，纵向切换）。
  - 这样在写作时就维护一份文档就可以了。即可以用来讲课，又可以做实践指导啥的。
 
- 缺点：
  - RStudio非常强__大__，只用来做这个有点笨重，非常浪费。学数学的朋友可以研究研究这个工具。
  - 没有直接预览。
  
# 2. 使用Markdown编写Slide

## 2.1. 基本流程

- Markdown本身的语法自行学习吧。
- revealjs
  - 建议阅读[revealjs官方文档]<https://github.com/hakimel/reveal.js>
- 重点：加separator分页就行了。
    - 默认veticalSeparator是“--”，Separator是“---”。
    - 在普通Markdown中通过添加Separator实现slide分页（分屏显示）, 前后加空行。
    - 在一级标题前加Separator, 一级标题内的用veticalSeparator。
    - 整体效果主是
      - 左右键会在一级标题级间切换；上下键可在一级标题内切换。
      - 第一行Slide都是一级标题，标题内的本级标题下的页面。

下面的图并未优化，略乱，只是为了展示排列效果。

![Slide排列效果，第一行为一级标题 ](./DR/Markdown_slides.png)

另一种利用横纵排列的思路是：

- 把自己想展示的内容放在第一行（横向排列），作为展示的主线。
- 一些细节或可放可不放的内容放在纵向排列，酌情展示。

## 2.2. 需要定制的方面

### 2.2.1. 文本段落左对齐

生成的SLIDE默认是居中对齐，看着非常怪。在文档前面加上如下内容就变成左对齐了。

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

下面的每一个“--”就会分出展示中的一页Slide.

```markdown

# 1 This is h1
--
### 1.1
--
### 1.2 
---
# 2 This is h1

```
# 4. 其他实现方式

这种方法现在很流行，有除了revealjs外，还有很多其他模板可以用。下面就使用pandoc，利用slidy模板生成slides。

> pandoc是实现文件格式转换的工具。本例中是把markdown格式，转换为html格式。

- Install  pandoc
    - **slidy** style works fine to me
        - pandoc yourfile.md -o yourslides.html -t slidy -s

具体参考下面的文献做可，实测可用。不怕命令行的可以用这种方式。转word，转PDF都可以。

This is the <https://linux.cn/article-4080-2.html> introduce how to produce slides with markdown.


