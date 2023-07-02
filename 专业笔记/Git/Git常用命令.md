# Git常用命令

***git就是免费的分布式版本控制系统***

 `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` 

![](pic\1.JPG)

git就像一个档案馆，***<u>记录你在哪个时候写了哪些代码</u>***

集中式的话最著名的是**svn**，区别：

- 集中式**只有一个公共档案馆**
- 分布式是**每人一个档案馆**，每个人在自己的版本里写代码，**如果自己写好了就可以和别人的版本进行合并**

github等就是git的一个**<u>托管平台</u>**

![](pic\0.png)

## 本地文件状态

![](pic\2.png)

## 远程仓库操作流程

> github已经禁止用户名密码，只能去官网生成一个token当密码来验证身份
>
> 也可以使用一个ssh来验证

![](pic\5.png)

## 分支

![](pic\6.png)

### git流模型

各个分支：

1. **feature**：用于**开发新功能**
2. **develop**：**收集各个feature分支，然后合并**，并测试
3. **release**：用于发布版本，并验证
4. **master**：发布版分支，最后把release分支合并到master分支，作为最后的发布版本
5. **hot fixes**：用于修复bug

![](pic\7.png)

## git init

初始化一个Git仓库： git init

## git log / git reflog

查看提交历史

## git clone  

**git clone xxx.git**

**git clone -b 分支名 *xxx.git***

## git branch

### 查看本地分支：

![](pic\8.png)

**git branch**

### 查看远程分支：

**git branch -r**

### 创建本地分支：

**git branch [name]** 

### 将branchA重命名为branchB：

 **git branch -m branchA branchB** 

### 删除分支

**git branch -d <name>**

### **合并某分支到当前分支**

**git merge <name>**

## git checkout 切换分支

![](pic\9.png)

### 创建新分支

**git checkout 分支名**

### 创建并切换到新分支

![](pic\11.png)

**git checkout -b  分支名  origin/分支名**

## git merge

![](pic\12.png)

### merge冲突

**两个分支中修改了相同的文件，导致不能merge**

![](pic\13.png)

**git status**：查看哪个文件出现了冲突

![](pic\14.png)

## git remote

### 查看远程仓库

git remote -v

### 添加远程仓库

git remote add [name] [url]

### 删除远程仓库

git remote rm [name]

### 修改远程仓库

git remote set-url --push [name] [newUrl]

## git pull

### 拉取远程仓库

 git pull [remoteName] [localBranchName]

## git push

### 推送远程仓库

git push [remoteName] [localBranchName]

git push -u 保存参数，下次直接git push就是一样的分支

![](pic\15.png)

## git fetch

***拉取远程仓库的分支***

两种跟踪远程仓库分支的方法：

![](pic\16.png)

## git add

###  新增的、修改的都加到缓存

**git add .**

### 新增、和修改的、和删除的都加到缓存

**git add -a*        

## git commit

***提交并且加注释*** 

![](pic\3.png)

git commit -am "init" ：其中am 意味着 add -m，***就是可以把没有暂存的文件也提交了***

![](pic\10.png)

## git status

***查看当前git文件状态信息***（查看是否有文件未提交）

![](pic\4.png)

## git diff / git log

查看文件具体更改和提交历史

## git stash

场景：突然master分支出现bug，你需要checkout切换到master分支，虽然也可以git commit提交写到一半的代码，但是不推荐这样做，所以**使用git stash来存储当前修改的代码**

![](pic\17.png)

stash apply 切换回分支：

![](pic\19.png)

## git reset

撤销之前的commit操作

- **--soft：只是撤销commit操作，暂存状态还是存在的**
- **--hard：不管取消暂存,之前修改的内容也取消了，彻底回到上次提交的状态（不建议使用）**

## git rebase

基线对齐，多个commit合并为一个commit