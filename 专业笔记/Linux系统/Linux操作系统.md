# Linux操作系统

# 常用命令

## tmux

tmux new -s s1	

tmux a

退出：ctrl + B +D

在使用tmux之前我们先了解关于tmux的几个名词：

- session，会话（任务）

- windows，窗口
- pane，窗格

关于session，很多人把session成为会话，但我觉得叫任务更适合一些。

在普通的终端中，窗口和其中由于session（任务）而启动的进程是连在一起的，关闭窗口，session就结束了，session内部的进程也会终止，不管是否运行完。但是在具体使用中，我们希望当前的session隐藏起来，在终端中做其他事情，但是又不希望session及其进程被关闭。这样就需要用到tmux，对session进行解绑。之后再想继续出来这个session的时候，再次绑定就可以回到之前的工作状态。

对于window可以理解为一个工作区，一个窗口。

对于一个session，可以创建好几个window，对于每一个窗口，都可以将其分解为几个pane小窗格。

### 1.安装

Ubuntu or Debian

```
sudo apt-get install tmux
```

CentOS or Fedora

```
sudo yum install tmux
```

Mac

```
brew install tmux
```

### 2.session操作

**启动**
新建session，可以在terminal上输入tmux命令，会自动生成一个id为0的session

```
tmux
```

也可以在建立时显式地说明session的名字，这个名字可以用于解绑后快速的重新进入该session：

```
tmux new -s your-session-name
```

**分离**
在tmux窗口中，按下**ctrl+b d**或者输入以下命令，就会将当前session与窗口分离，session转到后台执行：

```
tmux detach
```

**退出**
如果你想退出该session，可以杀死session：

```
tmux kill-session -t your-session-name
```

当然，也可以使用 **ctrl+d** 关闭该session的所有窗口来退出该session。

**绑定、解绑、切换session**
假设现在正处于session1，使用分离操作就是将session1进行解绑:

```
tmux detach
```

而如果你想再次绑定session1，可以使用命令：

```
tmux attach -t your-session-name
```

切换到指定session：

```
tmux switch -t your-session-name
```

**重命名session**

```
tmux rename-session -t old-session new-session
```

### 3.window操作

一个session可以有好几个window窗口。

**新建窗口**

```
tmux new-window
```

**新建一个指定名称的窗口**

```
tmux new-window -n your-window-name
```

**切换窗口**
命令就不记了，使用快捷键更方便：

- ctrl+b c: 创建一个新窗口（状态栏会显示多个窗口的信息）

- ctrl+b p: 切换到上一个窗口（按照状态栏的顺序）
- ctrl+b n: 切换到下一个窗口
- ctrl+b w: 从列表中选择窗口（这个最好用）

**重命名窗口**
执行：

```
tmux rename-window -t old_name new_name
```

**pane操作**
tmux可以将一个窗口分为几个窗格（pane），每个窗格运行不同的命令。

**划分窗格**

划分为上下两个窗格

```
tmux split-window
```

划分左右两个窗格

```
tmux split-window -h
```

其实划分窗格pane使用快捷键更方便，如果你当前pane正在运行程序不就没法使用命令了嘛。

- 左右划分：ctrl+b %

- 上下划分：ctrl+b "

**光标位置**

使用语句太麻烦了，使用快捷键最好：

ctrl+b arrow-key（方向键）：光标切换到其他窗格4.3 交换窗格位置

**当前窗格往上移**

```
tmux swap-pane -U
```

**当前窗格往下移**

```
tmux swap-pane -D
```

**关闭窗格**
ctrl+d，记住如果只有一个窗格就是关闭window哦

### 4.其他操作

列出所有快捷键，及其对应的 Tmux 命令

```
$ tmux list-keys
```

列出所有 Tmux 命令及其参数

```
$ tmux list-commands
```

列出当前所有 Tmux 会话的信息

```
$ tmux info
```

重新加载当前的 Tmux 配置

```
$ tmux source-file ~/.tmux.conf
```

tmux上下翻屏
使用快捷键ctrl+b [ ，就可以通过方向键上下移动使用PageUp和PageDown可以实现上下翻页

## 存储器

- **RAM：内存**
- **ROM：硬盘**

读取数据：**硬盘 ——>内存——>CPU**

![](PIC\0.png)

### 硬盘

**机械硬盘**

拆了的话很难恢复

- 采用**磁性碟片**来存储数据
- 盘面表面凹凸不平，**<u>凸起的地方是磁化，凹的地方是没被磁化的</u>**
- 凸起的代表1，凹的地方代表0
- 硬盘**根据转速来判断硬盘的好坏，<u>*磁盘转的越快，数据读取得越快*</u>**

如果磁头碰到了磁片，例如**运行时摔倒地上**，会产生很多的坏道，导致磁盘的坏死

一般硬盘**<u>不用的时候会自动降低转速，延长寿命</u>**

![1](PIC\1.png)

![](PIC\3.png)

存储数据时是按照**扇区为最小单位（4kb）来分配大小**的

![](PIC\4.png)

**固态硬盘**

采用**闪存颗粒（固态电子存储芯片阵列）**

![](PIC\2.png)

**如果硬盘采取顺序读写，速度甚至可以超越内存**，但日常生活中都是**<u>随机读写</u>**

![](PIC\5.png)

## 网络

ip地址是一种逻辑地址：ip地址 = **网络地址 + 主机地址** ，由4个字节构成

**子网掩码**的功能只有一个：**将IP地址划分为<u>网络地址和主机地址两部分</u>**

> 子网掩码判断两台计算机的**ip地址是否在同一个子网中**

网关：就是**网关设备的ip地址**，**<u>实现两个网络之间进行通讯和控制，用于连接外部网络</u>**

![](PIC\7.png)

**连网必需参数：**

- ip地址
- 子网掩码
- 默认网关
- DNS

![](PIC\6.png)

**DNS：域名解析服务**

C:\Windows\System32\drivers\etc目录下的**hosts文件**存储了ip和域名

> **域名劫持**：以前的钓鱼网站有**通过木马病毒修改hosts文件，然后让用户连接到自己的假网站**
>
> 现在**因为加密技术不会有域名劫持了**

![](PIC\8.png)

.com的域名是**全世界唯一**，.cn是中国独有

![](PIC\9.png)

### 连接模式

- **host-onboy**
- **bridged**
- **NAT**

虚拟机和主机连接主要分为两种：**桥接模式**和**NAT模式（网络地址转换模式）**

**桥接模式：**相当于**局域网中一台独立的主机**，虚拟系统和宿主机在同一子网下，相互可以访问，但缺点是**<u>如果虚拟机被配置了固定ip，可能会和别人的ip发生冲突</u>**

**NAT模式：**借用宿主机器的网络来访问公网，保证IP不会冲突，**<u>自己可以访问外面的机器，但外部的机器不能访问自己</u>**

![](PIC\11.png)

![](PIC\10.png)

## 软件分类

**系统软件（操作系统**）是**硬件和应用软件的<u>桥梁</u>**，这样只有它需要和硬件打交道

- **Windows：日常用户最多**
- **Linux：多用于服务器**
- **MacOS：苹果系统**

![](PIC\12.png)

**Android系统**是基于Linux内核开发的操作系统

![](PIC\13.png)

**GNU/Linux**

GNU是一个开源软件组织，**<u>世界上所有的软件应该开源免费</u>**，为了做一个模仿Unix的开源版本

作为操作系统，GNU的发展仍未完成，其中最大的问题是**<u>具有完备功能的内核尚未被开发成功</u>**。  GNU的内核，称为Hurd，是自由软件基金会发展的重点，但是其发展尚未成熟。在实际使用上，**多半使用Linux内核**、FreeBSD等替代方案，作为系统核心，其中主要的操作系统是Linux的发行版。

**GNU通用公共许可证简称为GPL**，是由自由软件基金会发行的用于计算机软件的协议证书，使用该证书的软件被称为自由软件。

- **给软件以版权保护。** 
- **给你提供许可证。它给你复制，发布和修改这些软件的法律许可。同样，为了保护每个作者和我们自己，我们需要清楚地让每个人明白，自由软件没有担保（no warranty）。如果由于其他某个人修改了软件，并继续加以传播。我们需要它的接受者明白：他们所得到的并不是原来的自由软件。由其他人引入的任何问题，不应损害原作者的声誉。**

> ​	Unix 系统被发明之后，大家用的很爽。但是后来**开始收费和商业闭源了**。一个叫 RMS 的大叔觉得很不爽，于是**<u>*发起 GNU 计划，模仿 Unix 的界面和使用方式，从头做一个开源的版本*</u>**。然后他自己做了**<u>编辑器 Emacs 和编译器 GCC。</u>**
>
> ​	GNU 是一个**计划或者叫运动**。在这个旗帜下成立了 FSF，**起草了 GPL** 等。
>
> ​	接下来大家**纷纷在 GNU 计划下做了很多的工作和项目，基本实现了当初的计划**。包括核心的 **gcc 和 glibc**。但是 GNU 系统**缺少操作系统内核。原定的内核叫 HURD，一直完不成**。同时 BSD（一种 UNIX 发行版）陷入版权纠纷，x86 平台开发暂停。然后一个叫 **<u>Linus 的同学为了在 PC 上运行 Unix，在 Minix 的启发下，开发了 Linux。注意，Linux 只是一个系统内核，系统启动之后使用的仍然是 gcc 和 bash 等软件。Linus 在发布 Linux 的时候选择了 GPL，因此符合 GNU 的宗旨。</u>**最后，大家突然发现，这玩意不正好是 GNU 计划缺的么。于是**合在一起打包发布叫 GNU / Linux**。然后大家念着念着省掉了前面部分，变成了 Linux 系统。实际上 Debian，RedHat 等 Linux 发行版中内核只占了很小一部分容量。

Linux发行版主要分为**两个体系**：

- **redhat**：Centos，产品免费，服务收费，**主要用于服务器**，没有用户界面
- **debian**：Ubuntu，kali等。主要给普通用户使用的桌面操作系统，**<u>有良好的视窗界面</u>**

> Linux发行版本虽然众多，但是真正属于**原始构建**的Linux版本可不多，只有少数几个，而大多数大家熟悉的或使用比较多的诸如CentOS，还有Ubuntu这一类属于**再构建版本**,简单来说就是这些版本是基于**原始构建版本**的基础之上再次修改及构建而来。
>
> 属于**原始构建版本**的真不多，我知道的只有以下几个：
>
> - **Redhat**，使用的是Yum/rpm包管理
> - **Debian**，使用的是Apt/deb包管理
> - **Arch Linux**，pacman包管理
>
> ### **Debian是Ubuntu的"老爸"**,Ubuntu是Debian的再构建版本
>
> ### Debian由社区负责，而Ubuntu由商业公司负责
>
> **<u>CentOS主流使用的版本还是6或7</u>**。CentOS 8 以及停更实际相当于是把 CentOS 改了一个名字——**CentOS Stream** ，这类似于好多开源项目在运作过程中也经常改名字。比如红帽 OpenShift 的社区版 origin 现在叫做 OKD。事实上，CentOS Stream 继承了 CentOS 的衣钵，与 CentOS 同源。我们认为这就是一个正常的版本迭代，或者说是社区项目的迭代

![](PIC\14.png)

## 虚拟化技术

![](PIC\15.png)

**虚拟化服务**：

更好利用计算机闲置资源，**和虚拟机和宿主机共享硬盘类似**

![](PIC\16.png)