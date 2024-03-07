# Git常用命令

***git就是免费的分布式版本控制系统***

> **集中式版本控制系统**
>
> 例如**SVN**
>
> 从**中央服务器**下载最新版本，然后修改后提交会中央服务器
>
> 优点：使用起来方便
>
> 缺点：中央服务器单点故障，所有人都无法工作
>
> ![](pic\20.png)

 `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` 

## 遇到的问题

问题：sp服务器上报错：ssh: Could not resolve hostname github.com: Temporary failure in name resolution
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.

解决方案：

![](pic\33.png)

问题：git pull的时候遇到冲突

解决方法：

![](pic\34.png)

```bash
 暂存到堆栈区
git stash

# 查看stash内容
git stash list
12345
```

要用到本地修改时，把`stash`内容应用到本地分支上：

```bash
git stash pop
```



## 工作区域

- **工作区**：自己电脑上的实际操作的目录
- **暂存区**：临时存储区域，临时保存即将提交的代码
- **本地仓库**：git存储代码和版本信息的位置，包含完整的项目历史和元数据

![](pic\21.png)

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



## 工作流模型

### GitGlow模型

***较为复杂的git分支工作流模型***

各个分支：

- **主分支：main，develop**
- **辅助分支：hotfix，feature，release**

![](pic\30.png)

1. **feature**：功能分支，用于**开发新功能**，功能开发完后即可合并回**开发分支**
2. **develop**：**开发分支**，**收集各个feature分支，然后合并**，用于**<u>开发</u>**
3. **release**：预发布分支，用于**<u>发布前的测试和验证</u>**，代码稳定后会***合并到主分支和开发分支，然后删除掉预发布分支***
4. **main/master**：**主线分支**，也叫发布版分支，包含项目最新稳定的代码，**<u>*不允许直接修改！只允许通过合并分支的方法来修改，每次合并都建议生成一个新的版本号*</u>**![](pic\29.png)
5. **hotfix**：修复分支，用于修复bug，完成任务后就可以删除分支

![](pic\7.png)

![](pic\32.png)

### GitHub Flow模型

***简单分支工作流模型***

**只有一个长期存在的主分支**，主分支设置了分支保护，禁止团队成员在主分支进行提交

团队成员从主分支中分离出自己的分支进行开发和测试，然后在本地分支提交代码，开发完成后发起**PR（pull ，request）**，团队成员可以对代码进行**<u>Review评审</u>**，如果没有问题就可以将这个PR发布和合并到主分支中，流程完成。

![](pic\31.png)

## .gitignore

忽略掉一些不该被加入到版本库中的文件，可以简化仓库

![](pic\23.png)

![](pic\24.png)

## git ls

**git ls-files** 可以查看暂存区中的内容

## git rm

git rm：将文件从工作区和暂存区中删除

> **记得还需要提交 git commit 一下，才能从版本库中删除**

## git init

初始化一个Git仓库： git init

## git log / git reflog

查看提交历史

git log --oneline 可以查看简洁版的提交记录

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

## git checkout 切换分支(为了避免歧义已经更改为git switch)

![](pic\9.png)

### 创建新分支

**git checkout 分支名**

### 创建并切换到新分支

![](pic\11.png)

**git checkout -b  分支名  origin/分支名**

## git merge

![](pic\12.png)

例如下图中将**dev分支的内容合并到main分支中**

![](pic\27.png)

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

注意git pull完会自动进行一次合并操作，如果有冲突就失败，git fetch只是拉取远程修改并不会自动合并

## git push

### 推送远程仓库

git push [remoteName] [localBranchName]

git push -u 保存参数，下次直接git push就是一样的分支

![](pic\15.png)

## git fetch

***拉取远程仓库的分支***

两种跟踪远程仓库分支的方法：

![](pic\16.png)

![](pic\26.png)

## git add

###  新增的、修改的都加到缓存

**git add .**

### 新增、和修改的、和删除的都加到缓存

git add -a

## git commit

***提交并且加注释*** 

![](pic\3.png)

git commit -am "init" ：其中-am 是 - a -m 意味着 add  + -m，***就是同时完成添加暂存和提交两个动作***

![](pic\10.png)

## git status

***查看当前git文件状态信息***（查看是否有文件未提交）

![](pic\4.png)

## git diff 

默认比较的是工作区和暂存区的内容，更改的文件和详细信息

git diff HEAD：比较工作区和版本库之间的差异

git diff 版本1ID 版本2ID：查看两个版本的差异内容

git diff 版本1ID HEAD：版本1和最新提交之间的差异

也可以加上文件名，表示只查看这个文件的差异

![](pic\22.png)

## git stash

场景：突然master分支出现bug，你需要checkout切换到master分支，虽然也可以git commit提交写到一半的代码，但是不推荐这样做，所以**使用git stash来存储当前修改的代码**

![](pic\17.png)

stash apply 切换回分支：

![](pic\19.png)

## git reset 回退版本

git中的所有操作都是可以回溯的

- **--soft：只是撤销commit操作，暂存状态还是存在的**，保留工作区和暂存区
- **--hard：不管取消暂存,之前修改的内容也取消了，彻底回到上次提交的状态（不建议使用）**
- **--mixed：保留工作区，丢弃暂存区**

当连续提交了多个版本，但可以**<u>合并为一个提交的时候，就可以利用soft和mixed来进行回退并重新提交</u>**

如果hard误操作了，也可以利用：

1. **git reflog**：查看误操作之前的版本号
2. **git reset --hard 版本号** ：即可回溯到那个版本号

## git rebase

***如果是只属于自己的分支，即确定只有自己在这个分支开发，且不希望提交历史太过复杂，建议使用rebase***

相比merge的

- 优点：**<u>不会新增额外的提交记录，形成线性历史，比较直观和干净</u>**
- <u>缺点：会改变提交历史，改变当前branch out的节点，**避免在共享分支使用**</u>

找到两个分支的共同祖先，把当前分支上从共同祖先到最新提交的所有提交移动到目标分支的最新提交后面

![28](pic\28.png)