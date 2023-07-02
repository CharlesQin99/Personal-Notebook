# Python3

## Anaconda

**目前国内外高校教学Python最流行的软件平台，是一个开源的Anaconda是专注于数据分析的Python发行版本。包含了Python的环境管理、代码编辑器、包管理等，一键安装方便快捷。**

Anaconda（[官方网站](https://link.zhihu.com/?target=https%3A//www.anaconda.com/download/%23macos)）就是可以便捷获取包且对包能够进行管理，同时对环境可以统一管理的发行版本。Anaconda包含了conda、Python在内的超过180个科学包及其依赖项。

### Anaconda和python区别

一个python环境中需要有一个**解释器**，和一个**包集合**

**1.安装包大小不同**

　　python自身缺少numpy、matplotlib、scipy、scikit-learn....等一系列包，**需要安装pip来导入这些包才能进行相应运算。**

　　Anaconda(开源的Python包管理器)是一个python发行版，包含了conda、Python等180多个科学包及其依赖项。包含了大量的包，使用Anaconda无需再去额外安装所需包。

**2.内置不同**

　　IPython是一个python的交互式shell，比默认的python shell好用得多，支持变量自动补全，自动缩进，支持bash shell命令，内置了许多很有用的功能和函数。而Anaconda Prompt是一个Anaconda的终端，可以便捷的操作conda环境。

## pip

pip是用于安装和管理软件包的包管理器

### pip 与 conda 比较

**→ 依赖项检查**

▪ pip：

① **不一定**会展示所需其他依赖包。

② 安装包时**或许**会直接忽略依赖项而安装，仅在结果中提示错误。

▪ conda：

**① 列出所需其他依赖包。**

**② 安装包时自动安装其依赖项。**

**③ 可以便捷地在包的不同版本中自由切换。**

> ▪ pip：维护多个环境难度较大。
>
> ▪ conda：比较方便地在不同环境之间进行切换，环境管理较为简单。
>
> ▪ pip：**仅适用于Python**。
>
> ▪ conda：适用于Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN。
>
> ▪ conda**结合**了**pip和virtualenv的功能。**

## virtualenv

virtualenv是用于创建一个**独立的**Python环境的工具。

▪ 解决问题：

1. 当一个程序需要使用Python 2.7版本，而另一个程序需要使用Python 3.6版本，如何同时使用这两个程序？如果将所有程序都安装在系统下的默认路径，如：***/usr/lib/python2.7/site-packages\***，当不小心升级了本不该升级的程序时，将会对其他的程序造成影响。
2. 如果想要安装程序并在程序运行时对其库或库的版本进行修改，都会导致程序的中断。
3. 在共享主机时，无法在全局 ***site-packages\*** 目录中安装包。

▪ virtualenv将会为它自己的安装目录创建一个环境，这并**不与**其他virtualenv环境共享库；同时也可以**选择性**地不连接已安装的全局库。

# EASY

## 1.两数之和

enumerate 通过字典来模拟哈希查询的过程

```python
nums=[2,7,9,10]

list(enumerate(nums))=[(0,2),(1,7),(2,9),(3,10)]

#常用：id,num in enumerate(nums) 
```

## 7.整数翻转

```python
if x>=0:ans=int(str(x)[::-1])
        else:ans=-int(str(-x)[::-1])
        if -2**31<=ans<=2**31-1:return ans
        return 0
    
知识点：
a=[1,2,3.4,5]
print(a)
[ 1 2 3 4 5 ]
 
print(a[-1]) #取最后一个元素
[5]
 
print(a[:-1])  #除了最后一个取全部
[ 1 2 3 4 ]
 
print(a[::-1]) #取从后向前（相反）的元素
[ 5 4 3 2 1 ]
 
print(a[2::-1]) #取从下标为2的元素翻转读取
[ 3 2 1 ]
```

## 9.回文数

```python
return str(x)==str(x)[::-1]

#解法2
class Solution:
     def isPalindrome(self, x: int) -> bool:
         b = list(str(x))
         b.reverse()
         # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
         b= ''.join(b)
         return str(x) == b
```

## 13.罗马数字转整数

```python
  d = {'I': 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        ans = 0
        ln = len(s)
        for i in range(ln):
            if i < ln - 1 and d[s[i]] < d[s[i + 1]]:
                ans -= d[s[i]]
            else:
                ans += d[s[i]]
        return ans
```

## 14.最长公共前缀

```python
def longestCommonPrefix(self, strs):
        s = ""
        for i in zip(*strs):
            ss = set(i)
            if len(ss) == 1:
                s += ss.pop()
            else:
                break  # 只要有一个不是一就跳出
        return s

#知识点：
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的list。
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
xyz = zip(x, y, z)

print xyz运行的结果是：
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

set（集合）
集合是一个无序不重复元素的集，其基本功能包括关系测试和消除重复元素。集合对象还支持 union（联合），intersection（交），difference（差）和sysmmetric difference（对称差集）等数学运算。
list1=[1,2,3,4,5]
list2=[3,4,5,6,7]
s1,s2=set(list1),set(list2)
list3=list(s1.intersection(s2))
print list3
得到两个set的交集

s1.union(s2)，用来得到两个set的并集
s1.difference(s2)  用来得到两个set的差集
s1.symmetric_difference(s2), 它相当于s1.union(s2)-s1.intersection(s2)。即s1和s2中不重复的元素的合集。
```

## 20.有效括号

```python
  dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False 
        return len(stack) == 1

```

## 21.合并有序链表

```python
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) {
            return l2;
        }
        if(l2 == null) {
            return l1;
        }

        if(l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

```

## 26.删除有序数组的重复项

```
	i = 0
    for j in range(len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1

```

## 27.移除元素

```python
    n=0
    for i in range(len(nums)):
        if nums[i]!=val:
            nums[n]=nums[i]
            n+=1
    return n
```

## 28.实现strStr()

```python
KMP，BM，Horspool，Sunday 算法。
```

## 35.搜索插入位置

```python
def searchInsert(self, nums: List[int], target: int) -> int:
        #巧方法
        nums.append(target)
        nums.sort()
        return nums.index(target)
```

## 53.最大子序和

```python
    def maxSubArray(self, nums: List[int]) -> int:
        max = nums[0]
        now = 0;
        for i in nums:
            if now <= 0:
                now = i
            else:
                now += i
            max = max if max>=now else now
        return max
```

## 58.最后一个单词的长度

```python
def lengthOfLastWord(self, s: str) -> int:
	return len(s.split()[-1])

>>> test1.rstrip()   ## 删除右边空格，r为right
>>> test1.lstrip()   ## 删除左边空格，l为left
>>> test1.strip()    ## 同时删除两边空格
```

## 66.加一

```python
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(1,len(digits)+1):
            digits[-i]+=1
            if digits[-i]==10:
                digits[-i]=0
            else:break			#模拟往前进位
        if digits[0]==0:digits=[1]+digits  #拼接数组
        return digits
```

## 67.二进制求和

```python
#利用bin与int将十进制与二进制互相转化即可
def addBinary(self, a: str, b: str) -> str:
    return bin(int(a,2)+int(b,2))[2:]

非内置
  r, p = '', 0
        d = len(b) - len(a)
        a = '0' * d + a
        b = '0' * -d + b		#将a和b补为一样的长度
        
        for i, j in zip(a[::-1], b[::-1]):
            s = int(i) + int(j) + p
            r = str(s % 2) + r
            p = s // 2
        return '1' + r if p else r

```

## 69.x的平方根

函数上任一点(x,f(x))处的切线斜率是2x。那么，x-f(x)/(2x)就是一个比x更接近的近似值。代入 f(x)=x^2-a得到x-(x^2-a)/(2x)，也就是(x+a/x)/2。

```python
  def mySqrt(self, x):
        if x == 0: return x
        x0, c = float(x), float(x)
        while True:
            xi = 0.5 * (x0 + c/x0)
            if abs(x0 - xi) < 1e-7: break
            x0 = xi
        return int(x0)
```

## 70.爬楼梯

```python
# 记忆化递归，自顶向下
def climbStairs(self, n: int) -> int:
    def dfs(i: int, memo) -> int:
        if i == 0 or i == 1:
            return 1
        if memo[i] == -1:
            memo[i] = dfs(i - 1, memo) + dfs(i - 2, memo)
        return memo[i]

    # -1 表示没有计算过，最大索引为 n，因此数组大小需要 n + 1
    return dfs(n, [-1] * (n + 1))

#自底向上dp
def climbStairs(self, n: int) -> int:
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]

#优化
def climbStairs(self, n: int) -> int:
    a = b = 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b

```

## 83.删除排序链表中重复元素

```python
 def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:return head
        n1=head
        n2=head.next
        while n2:
            if n1.val!=n2.val:
                n1.next=n2
                n1=n2
            n2=n2.next
        n1.next=None
        return head
    
    双指针
```

## 88.合并两个有序数组

```python
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[m:] = nums2
        nums1.sort()
```

## 94.二叉树的中序遍历

```python
  def inorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(tree, res):
            if tree == None:
                return
            dfs(tree.left, res)
            res.append(tree.val) # 中序访问该节点
            dfs(tree.right, res)
        res = []
        dfs(root, res)
        return res
```

## 100.相同的树

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:return True
        if not p or not q:return False
        if p.val != q.val:return False
        return self.isSameTree(p.left , q.left) and self.isSameTree(p.right , q.right)
```

## 101.对称二叉树

```python
 def isSymmetric(self, root: TreeNode) -> bool:
        if not root:return False
        def pan(left, right):
            if left is None and right is None:
                return True
            elif left is None or right is None:
                return False
            elif left.val != right.val:
                return False
            elif left and right:
                return pan(left.left, right.right) and pan(left.right, right.left)
        return pan(root.left, root.right)
```

## 104.二叉树的最大深度

```python
def maxDepth(self, root):
        if not root:
            return 0
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

## 108.将有序数组转换为二叉搜索树

```python
 def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def bin_build_tree(low,high):
            if low > high:
                return None
            mid = (low+high)//2
            root = TreeNode(nums[mid])
            root.left = bin_build_tree(low,mid-1)
            root.right = bin_build_tree(mid+1,high)
            return root
        return bin_build_tree(0,len(nums)-1)
```

## 110.平衡二叉树

```python
 def isBalanced(self, root: TreeNode) -> bool:
        return self.recur(root) != -1

    def recur(self, root):
        if not root: return 0
        left = self.recur(root.left)
        if left == -1: return -1
        right = self.recur(root.right)
        if right == -1: return -1
        return max(left, right) + 1 if abs(left - right) < 2 else -1
```

## 111.二叉树的最小深度

```python
def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        dq,depth = deque([root]), 1
        while dq:
            for i in range(len(dq)):
                tmp = dq.popleft()
                if not tmp.left and not tmp.right:
                    return depth
                if tmp.left:
                    dq.append(tmp.left)
                if tmp.right:
                    dq.append(tmp.right)
            depth += 1
        return depth
```

## 112.路径总和

```python
    def hasPathSum(self, root, sum):
        if not root: return False
        if not root.left and not root.right:
            return sum == root.val
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
```

## 118.杨辉三角

```python
特别解法：错一位再逐个相加
 def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 0: return []
        res = [[1]]
        while len(res) < numRows:
            newRow = [a+b for a, b in zip([0]+res[-1], res[-1]+[0])]
            res.append(newRow)      
        return res
平民解法：
 def generate(self, numRows: int) -> List[List[int]]:
        res = []
        for i in range(numRows):
            tmp = [1] * (i + 1)
            for j in range(1, i + 1):
                if j < i:
                    tmp[j] = res[i - 1][j - 1] + res[i - 1][j]
            res.append(tmp)
        return res
```

## 119.杨辉三角II

```python
def getRow(self, rowIndex: int) -> List[int]:
        res = [1]
        for i in range(1, rowIndex+1):
            res.append(1)
            for j in range(i-1, 0, -1):
                res[j] = res[j] + res[j-1]
        return res。
```

## 121.买卖股票的最佳时机

```python
 def maxProfit(self, prices: List[int]) -> int: 
        buy, sell = -float("inf"), 0
        for p in prices:
            buy = max(buy, 0 - p)
            sell = max(sell, buy + p)
        return sell
```

## 122.买卖股票的最佳时机II

```python
		profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0: profit += tmp
        return profit
```

## 123.买卖股票的最佳时机III

```python
 if len(prices) < 2:
       return 0
dp0 = 0             # 一直不买
dp1 = - prices[0]   # 到最后也只买入了一笔
dp2 = float("-inf") # 到最后买入一笔，卖出一笔
dp3 = float("-inf") # 到最后买入两笔，卖出一笔
dp4 = float("-inf") # 到最后买入两笔，卖出两笔

for i in range(1, len(prices)):
    dp1 = max(dp1, dp0 - prices[i])
    dp2 = max(dp2, dp1 + prices[i])
    dp3 = max(dp3, dp2 - prices[i])
    dp4 = max(dp4, dp3 + prices[i])
return dp4


核心：动态规划
```

## 125.验证回文串

```python
def isPalindrome(self, s: str) -> bool:
        ret = []
        for i in s:
            if i.isalnum():
                ret.append(i.lower())
        return ret == ret[::-1]
```

## 136.只出现一次的数字

```python
  def singleNumber(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1, len(nums)):
            res ^= nums[i]
        return res

#技巧：用异或
```

## 141.环形链表

```python
def hasCycle(self, head):
        if not head or not head.next:
            return False
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
   
#核心：快慢指针
```

## 144.二叉树的前序遍历

```python
  def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        res, stack = [], [root] # 利用栈进行临时存储
        while stack:
            node = stack.pop() # 取出一个节点，表示开始访问以该节点为根的子树
            res.append(node.val) # 首先访问该节点（先序），之后顺序入栈右子树，左子树
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
    
    
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(node):
            if not node:
                return
            res.append(node.val)
            dfs(node.left)
            dfs(node.right)
        
        res = []
        dfs(root)
        return res
```

## 145.二叉树的后序遍历

```python
def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        res, stack = [], [root]
        prev = root # 为了判断父子节点关系
        while stack:
            root = stack.pop() # 取出一个节点，表示开始访问以该节点为根的子树
            if (not root.left and not root.right) or (root.left == prev or root.right == prev): # 如果该节点为叶子节点，或者已经访问该节点的子节点
                res.append(root.val) # 直接访问
                prev = root
            else: # 否则就顺序把当前节点，右孩子，左孩子入栈
                stack.append(root)
                if root.right:
                    stack.append(root.right)
                if root.left:
                    stack.append(root.left)
        return res
思路：
1.尝试按顺序访问该节点的左右子树；
2.当左右子树都访问完毕，才可以访问该节点。

    def postorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            dfs(node.right)
            res.append(node.val)
        
        res = []
        dfs(root)
        return res
中序遍历:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
        
        res = []
        dfs(root)
        return res
```

## 155.最小栈

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]: 
            self.min_stack.append(x)
    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
    def top(self) -> int:
        return self.stack[-1]
    def getMin(self) -> int:
```

## 160.相交链表

```python
   def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a, b = headA, headB
        while a != b:
            if a:
                a = a.next 
            else:
                a = headB
            if b:
                b = b.next
            else:
                b = headA    
        return a
```

## 167.两数之和II-输入有序数组

```python
 def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left ,right = 0, len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right]  == target:
                return [left + 1, right + 1]
            if numbers[left] + numbers[right]  < target:
                left += 1
            if numbers[left] + numbers[right]  > target: 
                right -= 1
```

## 168.Excel表列名称

```python
 def convertToTitle(self, columnNumber: int) -> str:
        ans = []
        # 10进制 转换为 26进制，A对应1，B对应2,....Z对应26
        while columnNumber > 0:
            # 最右边位为取模运算的结果
            columnNumber -= 1
            # A的ASC码 65
            ans.append(chr(columnNumber%26 + 65))
            columnNumber //= 26				#" // " 表示整数除法,返回不大于结果的一个最大的整数
        return ''.join(ans[::-1])
```

## 169.多数元素

```python
#摩尔投票法:数目过半的数字与其他不过半的所有数字相减必然大于等于1，同时题目保证了只有一个解。
def majorityElement(self, nums):
        if not nums:
            return
        targetnum = nums[0]
        cnt = 0  
        for n in nums:
            if targetnum == n:
                cnt += 1
            else:
                cnt -= 1
            if cnt == -1:
                targetnum = n
                cnt = 0 
        return targetnum
```

## 171.Excel表列序号

```python
 def titleToNumber(self, columnTitle: str) -> int:
        i = 1
        cnt = 0
        for c in columnTitle[::-1]:
            cnt += (ord(c)-64)*i
            i*=26
        return cnt
```

## 172.阶乘后的零

```python
    def trailingZeroes(self, n: int) -> int:
        c5 = 5
        cnt = 0
        while (c5 <= n):
            cnt += n//c5
            c5*=5
        return cnt
```

## 175.组合两个表

```sql
select FirstName, LastName, City, State
from Person left join Address
on Person.PersonId = Address.PersonId;

用left join。因为需要展现所有人的数据，而Address里不一定有所有人（因为不一定所有人都有地址），但是Person里一定储存了所有人的数据。
最后一行得用on而不能用where。因为如果Address中查询的所有人的地址都存在，那么这么做没有问题。但是，题目中强调了，人一定存在，但地址不一定。一旦无法匹配到，where语句就行不通了。

1.on条件是在生成临时表时使用的条件，它不管on中的条件是否为真，都会返回左边表中的记录。
2、where条件是在临时表生成好后，再对临时表进行过滤的条件。这时已经没有left join的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉。
```

## 176.第二高的薪水

```sql
SELECT
    (SELECT DISTINCT
            Salary
        FROM
            Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1) AS SecondHighestSalary
        
 要点：
 SELECT * FROM table LIMIT 5;     //检索前 5 个记录行
SELECT * FROM table LIMIT 0,5;     //检索前 5 个记录行
SELECT * FROM table LIMIT 5,10;  // 检索记录行 6-15
#为了检索某行开始到最后的所有数据，可以设置第二个参数为-1
SELECT * FROM table LIMIT 95,-1; // 检索记录行 96-last
OFFSET用法：
SELECT * FROM table LIMIT 2 OFFSET 1;  //跳过1条数据读取2条数据，即读取2-3条数据
```

## 181.超过经理收入的员工

```sql
SELECT
     a.NAME AS Employee
FROM Employee AS a JOIN Employee AS b
     ON a.ManagerId = b.Id
     AND a.Salary > b.Salary
```

## 182.查找重复的电子邮箱

```sql
常规：
select Email from
(
  select Email, count(Email) as num
  from Person
  group by Email
) as statistic
where num > 1

使用having更加高效：
select Email
from Person
group by Email
having count(Email) > 1;
```

## 183.从不订购的客户

```sql
select a.Name as Customers
from Customers as a
left join Orders as b
on a.Id=b.CustomerId
where b.CustomerId is null;
```

## 190.颠倒二进制位

```python
 def reverseBits(self, n):
    res = 0
    for i in range(32):
    	res =res<<1 | n&1
    	n>>=1
    return res
```

## 191.位1的个数

```python
 def hammingWeight(self, n):
        return bin(n).count("1")
        
在 Python 语言中，使用 bin() 函数可以得到一个整数的二进制字符串。比如 bin(666) 会得到：
>>> bin(666)
'0b1010011010'
```

## 193.有效电话号码

```python
import re;
lines=open('file.txt').readlines();
lines=[i.strip()for i in lines];
ans=[i for i in lines if re.match('^\(\d{3}\) \d{3}-\d{4}$',i) or re.match('^\d{3}-\d{3}-\d{4}$',i)];
print('\n'.join(ans))

bash：
cat file.txt | grep -P "^(\([0-9]{3}\)\s|[0-9]{3}-)[0-9]{3}-[0-9]{4}$"
```

## 195.第十行

```bash
sed -n "10p" file.txt
cat file.txt|head -n 10|tail -n +10
```

## 196.删除重复的电子邮箱

```sql
DELETE p1 FROM Person p1,
    Person p2
WHERE
    p1.Email = p2.Email AND p1.Id > p2.Id
    
    
 DELETE p1       #只删除表p1
FROM Person AS p1
    INNER JOIN Person AS p2
WHERE p1.Email = p2.Email
    AND p1.Id > p2.Id
```

## 197.上升的温度

```sql
SELECT
    weather.id AS 'Id'
FROM
    weather
        JOIN
    weather w ON DATEDIFF(weather.date, w.date) = 1
        AND weather.Temperature > w.Temperature
        
重点：使用 DATEDIFF 来比较两个日期类型的值。
```

## 202.快乐数

```python
#方法1：快慢指针
def isHappy(self, n: int) -> bool:  
    def get_next(number):
        total_sum = 0
        while number > 0:
            number, digit = divmod(number, 10)
            #divmod(a, b) 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)
            total_sum += digit ** 2
        return total_sum

    slow_runner = n
    fast_runner = get_next(n)
    while fast_runner != 1 and slow_runner != fast_runner:
        slow_runner = get_next(slow_runner)
        fast_runner = get_next(get_next(fast_runner))
    return fast_runner == 1

#方法2：简单哈希
def isHappy(self, n: int) -> bool:

    def get_next(n):
        total_sum = 0
        while n > 0:
            n, digit = divmod(n, 10)
            total_sum += digit ** 2
        return total_sum

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1
```

## 203.移除链表元素

```python
#方法一：迭代法：
    def removeElements(self, head: ListNode, val: int) -> ListNode:
       if head is None:
            return head
        
        # removeElement方法会返回下一个Node节点
        head.next = self.removeElements(head.next, val)
        if head.val == val:
            next_node = head.next 
        else:
            next_node = head
        return next_node
    
    
   #方法二：虚拟头结点
   def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(-1)
        dummy.next = head
        p = dummy
        while p.next:
                if p.next.val == val:
                    p.next = p.next.next    #删除p.next
                else:
                    p = p.next
        return dummy.next
```

## 204.计数质数

```python
#方法一：埃拉托斯特尼筛
def count_primes_py(n):
    """
    求n以内的所有质数个数（纯python代码）
    """
    # 最小的质数是 2
    if n < 2:
        return 0

    isPrime = [1] * n
    isPrime[0] = isPrime[1] = 0   # 0和1不是质数，先排除掉

    # 埃式筛，把不大于根号 n 的所有质数的倍数剔除
    for i in range(2, int(n ** 0.5) + 1):
        if isPrime[i]:
            isPrime[i * i:n:i] = [0] * ((n - 1 - i * i) // i + 1)
    return sum(isPrime)

欧拉筛法的基本思想 ：在埃氏筛法的基础上，让每个合数只被它的最小质因子筛选一次，以达到不重复的目的。
```

## 205.同构字符串

```python
def isIsomorphic(self, s, t):
        for i in range(len(s)):
            if s.index(s[i]) != t.index(t[i]):
                return False
        return True
    
str.index(str, beg=0, end=len(string))
参数
str -- 指定检索的字符串
beg -- 开始索引，默认为0。
end -- 结束索引，默认为字符串的长度。

#方法2:哈希表
    def isIsomorphic(self, s, t):
        x = {}
        y = {}
        for i in range(len(s)):
            if (s[i] in x and x[s[i]] != t[i]) or (
                t[i] in y and y[t[i]] != s[i]):
                return False
            x[s[i]] = t[i]
            y[t[i]] = s[i]
        return True
```

## 206.反转链表

```python
 def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre
```

## 217.存在重复元素

```python
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums)!=len(set(nums))
```

## 219.存在重复元素II

```python
 哈希表做滑动数组+遍历
同样我们需要做的操作仍是遍历整个nums数组，但是遍历的每一步我们需要的操作要改换一下：
1. 我们构建一个哈希表window（之所以使用哈希表是因为我们需要对其进行很多的**【 查询】、【插入】和【删除】**操作，哈希表的这三个操作用时都是O(1)）
2. window的额定宽度是K+1，当 I>K时我们就需要将nusm[i-k-1]从哈希表中删除。
3. 查询当前值nums[i]是否存在哈希表中，若存在，则说明在k+1的范围内存在重复元素，return True
4. 将当前值nums[i]插入哈希表
5. 重复234操作直到遍历完整个nums
6. 若遍历完仍未找到，说明不存在这样的 [I,j]对，return False

    
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window={}
        for i in range(len(nums)):
            if i > k :
                window.pop(nums[i-k-1])
            if window and nums[i] in window:
                return True
            window[nums[i]]=1
        return False
         
```

## 225.用队列实现栈

```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.quene = []


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.quene.append(x)


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.quene.pop(-1)


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.quene[-1]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.quene)==0

```

## 226.翻转二叉树

```python
  def invertTree(self, root):
        if not root:
            return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


```

## 228.汇总区间

```python
 def summaryRanges(self, nums):
        nums.append(2 ** 32)
        ret, start = [], 0
        for i in range(1,len(nums)):
            if nums[i] - nums[i - 1] > 1:
                if i - 1 == start:
                    ret.append(str(nums[start]))
                else:
                    ret.append(f"{nums[start]}->{nums[i - 1]}")
                start = i
        return ret
```

## 231.2的幂22

```python
#位运算技巧
如果n为2的幂，恒有 n & (n - 1) == 0，这是因为：
n 二进制最高位为 1，其余所有位为 0；
n - 1 二进制最高位为 0，其余所有位为 1；

def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n - 1) == 0
```

## 232.用栈实现队列

```python
class MyQueue(object):

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)

    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self):
        return not self.stack1 and not self.stack2
```

## 234.回文链表

```python
    def isPalindrome(self, head: ListNode) -> bool:
        st = []
        while head:
            st.append(head.val)
            head = head.next
        return st == st[::-1]
```

## 235.二叉搜索树的最近公共祖先

```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```

## 242.有效的字母异位词

```python
 def isAnagram(self, s: str, t: str) -> bool:
        s1 = list(s)
        s1.sort()
        s1 = ''.join(s1)
        s2 = list(t)
        s2.sort()
        s2 = ''.join(s2)
        return s1 == s2
```

## 257.二叉树的所有路径

```python
    def getPaths(self, root, path, res):
        if not root:
            return
        # 节点值加入当前路径
        path += str(root.val)
        # 如果当前节点是叶子节点，将当前路径加入结果集
        if root.left == None and root.right == None:
            res.append(path)
        # 如果当前节点不是叶子节点，继续遍历左子树和右子树。
        else:
            path += '->'
            self.getPaths(root.left, path, res)
            self.getPaths(root.right, path, res)

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        res = []
        self.getPaths(root, '', res)
        return res
    
    
    #解法2 深度优先搜索
     def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(root:TreeNode,path:str,res:List[str]):
            if not root:
                return
            path += str(root.val)
            if not root.left and not root.right:
                res.append(path)
            else:
                path += '->'
                dfs(root.left,path,res)
                dfs(root.right,path,res)
            return res
        return dfs(root,'',[])
```

## 263.丑数

```python
def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False
        factors = [2, 3, 5]
        for factor in factors:
            while n % factor == 0:
                n //= factor
        return n == 1
```

## 278.第一个错误的版本

```python
#经典二分查找
def firstBadVersion(self, n: int) -> int:
        if n <= (2**31 - 1) and n >= 1:
            low = 0
            high = n -1 
            while low <= high :
                mid = (high-low)//2 + low  #//在Python中表示整数除法,返回大于结果的一个最大的整数,意思就是除法结果向下取整
                if isBadVersion(mid +1):
                    high = mid 
                    if low == high:
                        return low + 1
                else:
                    low = mid +1
                    if low == high:
                        return low + 1
```

## 283.移动零

```python
#解法1
def moveZeroes(nums):
	for i in range(nums.count(0)):
        nums.remove(0)
        nums.append(0)
        
#解法2
#快慢指针，通过交换两个指针对应的元素实现将0元素以到末尾。我们可以在列表开头放置两个指针left和right。right指针行移动。如果遇到非0数那么我们将left指针也向前移动使得left指针和right指针重合，那么交换两个指针对应的元素就是自己和自己交换不会又任何影响。如果遇到0元素则不移动left指针，但是right指针向前移动。那么left就会在right指针后一个位置。这时候交换两个指针对应的元素那么0元素就会和后一个非0元素互换位置实现0元素的移动。接着重复这一过程即可。

def moveZeroes(nums):
    right = 0
    left = 0
    while right < len(nums):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        right += 1
```

## 290.单词规律

```python
#两个哈希
def wordPattern(self, pattern: str, s: str) -> bool:
        temp = s.split(" ")
        m = len(pattern)
        n = len(temp)
        if m != n:
            return False
        hashmap1 = dict()
        hashmap2 = dict()
        for i in range(m):
            if pattern[i] in hashmap1:
                if hashmap1[pattern[i]] != temp[i]:
                    return False
            else:
                hashmap1[pattern[i]] = temp[i]
            if temp[i] in hashmap2:
                if hashmap2[temp[i]] != pattern[i]:
                    return False
            else:
                hashmap2[temp[i]] = pattern[i]
        return True
    
 #利用zip函数   
     def wordPattern(self, pattern: str, s: str) -> bool:
        return False if len(tmp := s.split()) != len(pattern) else len(set(zip(pattern, tmp))) == len(set(tmp)) == len(set(pattern))
```

## 338.比特位计数

```python
#解法1 用内嵌函数bin和count计数
def countBits(self, n: int) -> List[int]:
    res = []
    for i in range(n + 1):
        res.append(bin(i).count("1"))
    return res

#解法2 递归算法，偶数除以2后1的个数不变，奇数减1后1的个数减一
def countBits(self, num):
        res = []
        for i in range(num + 1):
            res.append(self.count(i))
        return res
    
    def count(self, num):
        if num == 0:
            return 0
        if num % 2 == 1:
            return self.count(num - 1) + 1
        return self.count(num // 2)
```

## 342.4的幂

```python
 #4的幂函数二进制表示只有一个1且0的个数是偶数，因为二进制是0bxxx所以0的个数会多一个标志0,所以0的个数必须是奇数
    def isPowerOfFour(self, n: int) -> bool:
         return bin(n).count('1') == 1 and bin(n).count('0') % 2 == 1 if n > 0 else False
```

## 345.翻转字符串中的元音字母

```python
 #双指针
    vowels = {'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U'}
    def reverseVowels(self, s: str) -> str:
        lt = list(s)
        l, r = 0, len(s) - 1
        while l < r:
            if lt[l] in self.vowels and lt[r] in self.vowels:
                lt[l], lt[r] = lt[r], lt[l]
                l += 1
                r -= 1
            elif lt[l] in self.vowels:
                r -= 1
            else:
                l += 1
        return ''.join(lt)
```

## 349.两个数组的交集

```python
    #哈希表是查找中非常高效的方法
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums1 or not nums2:
            return []

        # 初始化哈希表
        hash = {}
        # 初始化结果列表，存放最后结果
        res = []

        # 哈希表 key 为 nums1 数组中的数，value 为值
        for i in nums1:
            if not hash.get(i):
                hash[i] = 1
        # 遍历 nums，如果 nums2 数组中的数出现在哈希表中，对应数放入结果列表，对应 value 值置-为0
        for j in nums2:
            if hash.get(j):
                res.append(j)
                hash[j] = 0

        return res
    
    
    #方法2 
     def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        a = set(nums1)
        b = set(nums2)
        theList = []
        if len(a) < len(b):
            for num1 in a:
                if num1 in b:
                    theList.append(num1)
        elif len(a) >= len(b):
            for num2 in b:
                if num2 in a :
                    theList.append(num2)
        
        return theList

```

## 350. 两个数组的交集II

```python
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:   
        nums1.sort()
        nums2.sort()
        left,right = 0, 0
        ans = []

        while left < len(nums1) and right < len(nums2):
            if nums1[left] == nums2[right]:
                ans.append(nums1[left])
                left += 1
                right += 1
            elif nums1[left] < nums2[right]:
                left += 1
            else:
                right += 1
        return ans
```

## 367.有效完全平方数

```python
 #二分查找法 
 def isPerfectSquare(self, num: int) -> bool:
        l, r = 1, num
        while l < r:
            mid = (l + r) // 2
            if mid * mid < num:
                l = mid + 1
            else:
                r = mid
        return l * l == num
    
#等差数列法
有一个公式
1 + 3 + 5 + 7 + ... (2N-1)= N^2
所以任意一个平方数可以表示成这样的奇数序列和。

class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        i = 1
        while num > 0:
            num -= i
            i += 2
        return num == 0
    
    
 #牛顿迭代法 数学方法求解 
def isPerfectSquare(self, num: int) -> bool:
        i = num
        while i * i > num:
            i = (i + num / i) // 2
        return i * i == num
```

## 374.猜数字大小

```python
   #经典二分查找
   def guessNumber(self, n: int) -> int:
        left ,right = 1, n
        while left < right:
            mid = (right-left)//2 + left
            if guess(mid) == 0:
                return mid
            elif guess(mid) == -1:
                right = mid
            elif guess(mid) == 1:
                left= mid + 1
            print(mid)
        return left
```

## 383.赎金信

```python
    #哈希查找
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        hash = {}
        for i in list(magazine):
            if hash.get(i):
                hash[i] += 1
            else:
                hash[i] = 1

        print(hash)
        for i in list(ransomNote):
            if hash.get(i):
                hash[i] -= 1
                if hash[i] < 0:
                    return False
            else:
                return False
        return True
```

## 387.字符串中的第一个唯一字符

```python
#哈希查找
def firstUniqChar(self, s: str) -> int:
        hashtable = {}

        #将s列表里的element储存为key，index储存为value
        for index,string in enumerate(s):
            if string in hashtable:
                hashtable[string] = -1
            else:
                hashtable[string] = index

        for key in hashtable:
            if hashtable[key] != -1:
                return hashtable[key]
        return -1
```

## 389.找不同

```python
  #计数法
    def findTheDifference(self, s: str, t: str) -> str:
        str_count = [0] * 26
        for ch in s:
            str_count[ord(ch) - ord('a')] += 1
        for ch in t:
            str_count[ord(ch) - ord('a')] -= 1
            if str_count[ord(ch) - ord('a')] < 0:
                return ch
            
 #利用位运算
    def findTheDifference(self, s: str, t: str) -> str:
        mask = 0
        for char in s+t:
            mask ^= ord(char)
        return chr(mask)
```

## 392.判断子序列

```python
 #双指针
 def isSubsequence(self, s: str, t: str) -> bool:
        st, tt = 0, 0
        while st < len(s) and tt <len(t):
            if s[st] == t[tt]: 
                st += 1
                tt += 1
            else:
                tt += 1
        return st == len(s)
```

## 404.左叶子之和

```python
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        res = []
        def dfs(root):
            if not root:return
            if root.left and not root.left.left and not root.left.right:
                res.append(root.left.val)
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return sum(res)
```

## 405.数字转换为16进制

```python
 def toHex(self, num: int) -> str:
        CONV = "0123456789abcdef"
        ans = []
        # 32位2进制数，转换成16进制 -> 4个一组，一共八组
        for _ in range(8):
            # 当输入值num为-1 ，第一次进入循环
            ans.append(num % 16)  # num % 16 = 15
            num //= 16  # num // 16 = -1
            # Python中的//运算取整除：向下取接近商的整数
            # %取模运算返回整除的余数 （余数 = 被除数 - 除数 * 商）
            # 负整数 // 正整数 的最大值为-1
            #   -1 // 16 = -1
            #   -1 % 16 = 15
            #   即如num为负数，则一定会跑满for的8次循环
            # 正整数 // 正整数 的最小值为0
            #   1 // 16 = 0
            #   1 % 16 = 1
            #   即num为正数时，有可能触发下面的if语句，提前结束for循环
            if not num:  # 如果num不为0则继续下一次for循环
                break  # 如果num为0则终止for循环
            # 正整数 // 负整数 的最大值为-1，如1 // -16 = -1; 1 % -16 = -15
        return "".join(CONV[n] for n in ans[::-1])
```

## 409.最长回文串

```python
   # 使用Counter s = "abccccdd"  执行Counter(s) 后得到('a', 1)('b', 1)('c', 4)('d', 2)
    def longestPalindrome(self, s: str) -> int:
        b,d=Counter(s), 0
        for i in b.values():
            if i!=1:
                d+=i//2*2
        if d!=len(s):
            d+=1
        return d

```

## 414.第三大的数

```python
  # sort(reverse=True) 从大到小排序
    def thirdMax(self, nums: List[int]) -> int:
        st = list(set(nums))
        st.sort(reverse=True)
        if len(st) < 3:
            return st[0]
        else:
            return st[2]
```

## 415.字符串相加

```python
    #大数相加
    def addStrings(self, num1: str, num2: str) -> str:
        res = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            tmp = n1 + n2 + carry
            carry = tmp // 10
            res = str(tmp % 10) + res
            i, j = i - 1, j - 1
        return "1" + res if carry else res
    
```

## 448.找到所有数组中消失的数字 

```python
''' 
difference() 方法用于返回集合的差集，即返回的集合元素包含在第一个集合中，但不包含在第二个集合(方法的参数) set.difference(set)
'''

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        mat = [i+1 for i in range(len(nums))]
        return list(set(mat).difference(set(nums)))
```

## 453.最小操作次数使数组元素相等

```python
1. 假设我们最少的操作次数是k,k次操作后每个元素相等，相等元素设为target
2. 对于整个列表的n - 1个元素都要进行加一操作那么增加的总数是 k * (n - 1)
3. 原本的列表之和为 sum(nums)，k次操作后应该存在这样的关系等式：
k[最少操作次数] * (n - 1)[每次对n - 1个元素进行操作] + sum(nums)[原列表的和] = target[操作后的相等元素] * n
target的值是 k + nums中的最小值，求解方程即可

 def minMoves(self, nums: List[int]) -> int:
     return sum(nums) - len(nums) * min(nums) if len(nums) != 1 else 0

```

## 455.分发饼干

```python
 #排序后双指针
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        gi ,si = 0, 0
        g.sort()
        s.sort()
        ans = 0
        glen,slen = len(g), len(s) 
        while gi <glen and si<slen:
            if s[si] >= g[gi]:
                ans += 1
                si += 1
                gi += 1
            else:
                si += 1
        return ans
```

## 459.重复的子字符串

```python
   # s = s[-1] + s[:-1] 完成字符串右移操作
   def repeatedSubstringPattern(self, s: str) -> bool:
        ans = 0
        tmp = s
        if len(s) == 1: return False
        while ans < len(s)//2:
            #s.insert(0, s.pop()) 对数组使用
            s = s[-1] + s[:-1]
            if tmp == s:
                return True
            ans += 1
        return False
```

## 461.汉明距离

```python
  #bin(x).count('1')计算二进制表示中的1的个数
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x^y).count('1')
```

## 463.岛屿的周长

```python
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        rowlen = len(grid)
        if not grid or rowlen == 0:
            return 0
        collen = len(grid[0])
        res = 0
        for i in range(rowlen):
            for j in range(collen):
                if grid[i][j] == 1:
                    res += 4
                    if i-1>=0 and grid[i-1][j] == 1:
                        res -= 2
                    if j-1>=0 and grid[i][j-1] == 1:
                        res -= 2
        return res
```

## 476.数字的补数

```python
num = 5
bin(5) = 'ob101'
len('ob101')-2 = 3
2**3 - 1 = 7
#最后求异或即可

class Solution:
    def findComplement(self, num: int) -> int:
        return num^(2**(len(bin(num))-2)-1)
```

## 482.密钥私有化

```python
#字符串大小写常用操作
str.lower() #将字符串中的大写字母转换成小写字母
str1.upper() #将字符串的小写字母转换为大写字母
strl.capitalize() #将字符串的第一个字母变成大写，其余字母变为小写 Happy new year
str3.title()   #所有英文单词首字母大写，其余英文字母小写 I Love Python
str3.swapcase  #将字符串str中的大小写字母同时进行互换

    def licenseKeyFormatting(self, s: str, k: int) -> str:
        s = s.replace('-','')
        s = s.upper()
        slen = len(s)
        res = ''
        a = slen % k
        b = slen // k
        if a != 0:
            pass
        count = 0
        for i in range(slen):
            res += s[i]
            count += 1
            if i == slen -1:
                break
            if i + 1 == a or count == k:
                res += '-'
                count = 0
        return res
```

## 485.最大连续1的个数

```python
  # sl = "".join(str) 字符数组转字符串
  # sl = "".join(str(i) for i in nums) 数字数组转字符串
  def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        #sl = "".join(str)
        sl = "".join(str(i) for i in nums)
        max= 0
        for i in sl.split('0'):
            if len(i) > max: max = len(i)
        return max
```

## 492.构造矩形

```python
	def constructRectangle(self, area: int) -> List[int]:
        w = int(sqrt(area))
        while area % w:
            w -= 1
        return [area // w, w]
    
    #另一种写法
    #range(start, stop[, step])  stop结束但不包括stop，step为步长，可以为负数
      for i in range(int(area**0.5)+1,-1,-1):
            if area % i == 0:
                return [area//i,i] if  area//i >= i else [i,area//i]
```

## 495.提莫攻击

```python
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        ans = 0 
        index = 0
        if len(timeSeries) == 0: return 0
        while index + 1 < len(timeSeries):
            if timeSeries[index + 1] - timeSeries[index] < duration:
                ans += timeSeries[index + 1] - timeSeries[index]
            else:
                ans += duration
            index += 1
        return ans + duration
```

## 496.下一个更大的元素

```python
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        window, d = [], dict()
        for num in nums2:
            while window and window[-1] < num:
                small = window.pop()
                d[small] = num
            window.append(num)
        return [d[num] if num in d else -1 for num in nums1]
```

## 500.键盘行

```python
  #lambda表达式  any函数中元素全为0或者''或者FALSE时，返回FALSE，否则返回TRUE
    def findWords(self, words: List[str]) -> List[str]:
        sets = (set("qwertyuiop"), set("asdfghjkl"), set("zxcvbnm"))
        return list(filter(lambda x: any(set(x.lower()).issubset(y) for y in sets), words))
    
   #普通解法
    def findWords(self, words: List[str]) -> List[str]:
    	return [word for word in words if any(set(word.lower()).issubset(y) for y in sets)]
```

## 501.二叉搜索树中的众数

```python
#二叉搜索树的中序遍历

class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if not root:
            return
        result = []
        max_count =  count = 0 
        base = float("-inf") 
        #对当前值进行处理
        def update(x):
            nonlocal max_count,count,base,result
            #相等则计数+1
            if x == base:
                count += 1
            else:
                #不相等，说明该数的节点已经全部遍历完成，更新base，count为1
                base = x
                count = 1
            #计数如果等于max，则加入result
            if count == max_count:
                result.append(base)
            #计数如果大于max，则要重置result，并把该值加入      
            if count > max_count:
                max_count = count
                result = []
                result.append(base)
        #二叉查找树，中序遍历，数据从小到大顺序处理
        def mid_order(root):
            if root:
                nonlocal max_count,count,base,result
                mid_order(root.left)
                update(root.val)
                mid_order(root.right)
        mid_order(root)
        return result
```

## 504.七进制数

```python
字符串与数字的转换
字符串str转换成int: int_value = int(str_value)
int转换成字符串str: str_value = str(int_value)
print(ord("A"))  # 打印结果为65
print(chr(65))  # 打印结果为A

    def convertToBase7(self, num: int) -> str:
        tmp = num
        num = abs(num)
        ans = []
        if num == 0: return "0"
        while num:
            ans.append(num%7)
            num //= 7
        ts = ''.join(str(n) for n in ans[::-1])
        return ts if tmp >= 0 else "-" + ts 
```

## 506.相对名词

```python
'''
数组拷贝
b = a   浅拷贝  等号赋值实际上只是引用地址的传递，b会随着a的改变而改变
b = a[:] 深拷贝  这样就是两个独立数组
'''

def findRelativeRanks(self, score: List[int]) -> List[str]:
        dict= {}
        ans = []
        t_score = score[:]    #注意数组拷贝方法
        score.sort(reverse = True)
        for i in range(len(score)):
            if i == 0: dict[score[i]] = "Gold Medal"
            elif i == 1: dict[score[i]] = "Silver Medal"
            elif i == 2: dict[score[i]] = "Bronze Medal"
            else:
                dict[score[i]] = str(i+1)
        for j in t_score:
            ans.append(dict[j])
        return ans
```

## 507.完美数

```python
 #暴力法
    def checkPerfectNumber(self, num: int) -> bool:
        if num == 1:
            return False
        #计数从1开始
        total = 1
        #我们只需要判断`2-int(sqrt(num))+1`的数，全部累加
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                #这里total要加上i和num // i
                total += (i + num // i)
        return total == num
    
 #数论：欧几里得定理：完美数: 2^{p - 1} * (2^{p} - 1) 且 p和2^{p}-1都是质数 
    def checkPerfectNumber(self, num: int) -> bool:
       #判断是否为质数的函数
    	def isPrime(x):
            for j in range(2, x//2):
                if not x % j:
                    return False
            return x > 1

        # 2^(p-1) * (2^p - 1)
        p = 1
        while not num % 2:
            num //= 2
            p += 1
        return num + 1 == pow(2, p) and isPrime(p) and isPrime(num)
    
 #打表法
    def checkPerfectNumber(self, num: int) -> bool:
        return num in set(6, 28, 496, 8128, 33550336)
```

## 509.斐波那契数列

```python
    def fib(self, n: int) -> int:
        if not n:
            return 0
        ans = [0,1]
        i = 2
        while i <= n:
            ans.append(ans[i-2] + ans[i-1])
            i += 1
        return ans[-1]
```

## 520.检测大写字母

```python
s.isalnum()   所有字符都是数字或者字母，为真返回 Ture，否则返回 False。（重点，这是字母数字一起判断的！！）
s.isalpha()   所有字符都是字母，为真返回 Ture，否则返回 False。（只判断字母）
s.isdigit()   所有字符都是数字，为真返回 Ture，否则返回 False。（只判断数字）
s.islower()   所有字符都是小写，为真返回 Ture，否则返回 False。
s.isupper()   所有字符都是大写，为真返回 Ture，否则返回 False。
s.istitle()   所有单词都是首字母大写，为真返回 Ture，否则返回 False。
s.isspace()   所有字符都是空白字符，为真返回 Ture，否则返回 False。

def detectCapitalUse(self, word: str) -> bool:
	return word.islower() or word.isupper() or word.istitle()
```

## 521.最长的特殊序列I

```python
    def findLUSlength(self, a: str, b: str) -> int:
         return -1 if a == b else max(len(a), len(b))
```

## 530.二叉树的最小绝对差

```python
   #中序遍历
    def getMinimumDifference(self, root: TreeNode) -> int:
        stack = []
        cur = root
        pre = None
        result = float('inf')

        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre:
                    result = min(result, cur.val - pre.val) 
                pre = cur
                cur = cur.right
        return result
    
    #中序遍历第二种写法
    def getMinimumDifference(self, root: TreeNode) -> int:
        st = []
        p = root
        pre = -float('inf')
        min_val = float('inf')
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            cur = p.val
            if cur - pre < min_val:
                min_val = cur - pre
            pre = cur
            p = p.right
        return min_val
```

## 541.反转字符串II

```python
	#range的步长技巧 而且list不用担心越界问题，arr[i,i+k]超过数组下标默认截取到最后
	def reverseStr(self, s: str, k: int) -> str:
        n=len(s)
        arr=list(s)
        for i in range(0,n,2*k):
            arr[i:i+k]=arr[i:i+k][::-1]
        return "".join(arr)
```

## 543.二叉树的直径

```python
采用分治和递归的思想：
    根节点为root的二叉树的直径 = max(root->left的直径，root->right的直径，root->left的最大深度+root->right的最大深度+1)
    
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans = 1
        def depth(root):
            if not root: return 0
            L = depth(root.left)
            R = depth(root.right)
            self.ans = max(self.ans, L + R + 1)
            return max(L, R) + 1
        depth(root)
        return self.ans - 1
```

## 551.学生出勤记录

```python
 Python判断字符串是否包含指定字符串的方法 字符串查找
  方法一：使用操作符 in          flag = re in str     在的话flag返回True
  方法二：使用string模块的函数
  1. find()
  检测字符串中是否包含子字符串，如果指定 beg（开始） 和 end（结束） 范围内，则检查是否包含在指定范围内，如果包含子字符串，返回第一次出现该子字符串的开始索引值，否则返回 -1
  2. rfind()
  用法和上述一致，只是这个返回最后出现的的开始索引值。
  3. index()
  和find()方法一样，如果指定范围内包含指定字符串，返回的是索引值在字符串中的起始位置,只不过如果str不在 string中会报一个异常。
  4. rindex()
  和rfind()方法一样，返回指定字符串 str 在字符串中最后出现的位置，只不过如果str不在 string中会报一个异常。
  
def checkRecord(self, s: str) -> bool:
    st = "LLL"
    return s.count("A") < 2 and not st in s
    #return s.count("A") < 2 and s.find(st) == -1
```

## 557.反转字符串中的单词

```python
def reverseWords(self, s: str) -> str:
    sl = s[::-1]
    st = sl.split(' ')
    return ' '.join(st[::-1])
```

## 559.N叉树的最大深度

```python
def maxDepth(self, root: 'Node') -> int:
    if not root:
        return 0
    d = 1
    s = [(root, 1)]
    while s:
        cur, h = s.pop()
        for child in cur.children:
            s.append((child, h + 1))
            d = max(d, h)
    return d
```

## 561.数组拆分

```python
def arrayPairSum(self, nums: List[int]) -> int:
    nums.sort()
    ans = 0
    for i in range(0,len(nums),2):
        ans += nums[i]
    return ans
```

## 563.二叉树的坡度

```python
def findTilt(self, root: Optional[TreeNode]) ->int:
    ans = []
    def sum_tree(root):
        if not root: return 0
        L = sum_tree(root.left)
        R = sum_tree(root.right)
        ans.append(abs(L-R))
        return L + R + root.val
    sum_tree(root)
    return sum(ans)
```

## 566.重塑矩阵

```python
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        if len(mat) * len(mat[0]) != r * c:
            return mat
        result = []
        path = []
        for words in mat:
            for word in words:
                path.append(word)
                if len(path) == c:
                    result.append(path)
                    path = []
        return result
```

## 572.另一棵树的子树

```python
#判断二叉树t是否为s的子树
    def isSubtree(self, s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        return self.isSameTree(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
  
#判断两棵二叉树是否相同
    def isSameTree(self, s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        return s.val == t.val and self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)
```

## 575.分糖果

```python
def distributeCandies(self, candyType: List[int]) -> int:
    return min(len(candyType)//2, len(set(candyType)))
```

## 589.N叉树的前序遍历

```python
def preorder(self, root: 'Node') -> List[int]:
    stack, ans = [root], []
    while stack:
        node = stack.pop()
        if node:
            ans.append(node.val)
            for child in node.children[::-1]:
                stack.append(child)
    return ans
```

## 590.N叉树的后序遍历

```python
def postorder(self, root: 'Node') -> List[int]:
    ans = []
    def dfs(node):
        if node is None:
            return
        for ch in node.children:
            dfs(ch)
        ans.append(node.val)
    dfs(root)
    return ans
```

## 594.最长的和谐子序列

```python
def findLHS(self, nums: List[int]) -> int:
    res = 0
    nums2 =set(nums)
    for i in nums2:
        son = nums.count(i-1)
        if son !=0:
            res = max(res,nums.count(i)+son)
    return res
```

## 598.范围求和

```python
#data = list(zip(*data)) 每列的数据组成一个元组 , 再一起组成一个列表
def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
    zl = list(zip(*ops))
    if not len(ops): return m*n
    if len(zl) == 1:
        return ops[0][0] * ops[0][1]
    return min(zl[0]) * min(zl[1])
```

## 599.两个列表的最小索引总和

```python
#index函数用来数组中查找某个元素位置
def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
    s1 = set(list1)
    s2 = set(list2)

    sun = s1.intersection(s2)
    mv = float('inf')
    ans = []

    for i in sun:
        index1 = list1.index(i)
        index2 = list2.index(i)
        tp = index1 + index2
        if tp < mv:
            mv = tp
            ans.clear()
            ans.append(i)
        elif tp == mv:
            ans.append(i)
    return ans
```

## 605.种花问题

```python
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    ans = 0
    if len(flowerbed) == 0:
        return False
    flowerbed.insert(0, 0)
    flowerbed.append(0)

    for i in range(1,len(flowerbed)-1):
        if flowerbed[i] == 0 and flowerbed[i-1] != 1 and flowerbed[i+1] != 1:
            flowerbed[i] = 1
            n -= 1
    return n<=0
```

## 606.根据二叉树创建字符串

```python
def tree2str(self, root: Optional[TreeNode]) -> str:
    if not root:
        return ''

    res = str(root.val)  
    if not root.left and not root.right:
        return res

    left_res = self.tree2str(root.left)
    right_res = self.tree2str(root.right)

    res += '(' + left_res + ')'
    if right_res: 
        res += '(' + right_res + ')'
    return res
```

## 617.合并二叉树

```python
def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
    def dfs(r1, r2): #定义一个递归函数
        if not r1 or not r2: #递归的结束条件是r1或者r2中的一个为空，如果r1为空，返回r2，如果r2为空，返回r1
            return r1 if r1 else r2
        root = TreeNode()  #初始化一颗新的树
        root.val = r1.val + r2.val #如果r1和r2都不为空，返回二者相加
        root.left = dfs(r1.left, r2.left) #对两棵树的左子树继续递归
        root.right = dfs(r1.right, r2.right)#对两棵树的右子树继续递归
        return root     #返回相加后的树
    
    return dfs(root1, root2)#返回执行后的递归子函数
```

## 629.三个数的最大乘积

```python
def maximumProduct(self, nums: List[int]) -> int:
    nlen = len(nums)    
    if nlen == 3:
        return nums[0] * nums[1] * nums[2]
    nums.sort()
    return max(nums[0]*nums[1]*nums[-1],nums[-1]*nums[-2]*nums[-3])
```

## 637.二叉树的层平均值

```python
def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
    if root == None:
        return []
    res = []
    queue = [root]
    while queue:
        # 存储当前层的孩子节点列表
        childNodes = []
        # 存储当前层的节点数
        cnt = len(queue)
        # 存储当前层节点和
        sum = 0
        # 求当前层节点和
        for node in queue:
            sum += node.val
        res.append(sum/cnt)
        
        for node in queue:
            # 若节点存在左孩子，入队
            if node.left:
                childNodes.append(node.left)
            # 若节点存在右孩子，入队
            if node.right:
                childNodes.append(node.right)
        # 更新队列为下一层的节点，继续遍历
        queue = childNodes
    return res
```

## 643.子数组最大平均值

```python
def findMaxAverage(self, nums: List[int], k: int) -> float:
    max = sum(nums[:k])
    sm = max
    for i in range(k,len(nums)):
        sm = sm + nums[i] - nums[i-k]
        if sm > max:
            max = sm
    return max/k
```

## 645.错误的集合

```python
def findErrorNums(self, nums: List[int]) -> List[int]:
	nums.sort()
    l = len(nums)
    minus = l*(l+1)//2 - sum(nums)

	for i in range(len(nums)):
		if nums[i] == nums[i+1]:
			return [nums[i], nums[i] + minus]
	return []

#运用Counter实现元素计数  most_common()中间的参数为出现频率前几多的元素
def findErrorNums(self, nums: List[int]) -> List[int]:
    check = Counter(nums)
    print(check)
    a = check.most_common(1)
    print(a)
```

## 653.两数之和

```python
def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
    s = set()

    def dfs(root):
        if not root:
            return False
        else:
            if k - root.val in s:
                return True
            s.add(root.val)
            if dfs(root.left) or dfs(root.right):
                return True
        return False
    
    return dfs(root)
```

## 657.机器人能否返回原点

```python
def judgeCircle(self, moves: str) -> bool:
    uc = moves.count("U")
    dc = moves.count("D")
    rc = moves.count("R")
    lc = moves.count("L")

    return uc == dc and rc == lc
```

## 661.图片平滑器

```python
def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
    nimg = [[0 for j in i] for i in img]
    n, m = len(img), len(img[0])
    for x in range(n):
        for y in range(m):
            nimg[x][y] += sum(img[i][j] for i in range(max(0, x - 1), min(n, x + 2)) for j in range(max(0, y - 1), min(m, y + 2))) // ((min(n, x + 2) - max(0, x - 1)) * (min(m, y + 2) - max(0, y - 1)))
    return nimg
```

## 671.二叉树中的第二小的节点

```python
#不完全深搜
def findSecondMinimumValue(self, root: TreeNode) -> int:
    def find_sec(node):
        if not node:
            return float('inf')
        if node.val!=root.val:
            return node.val
        else:
            return min(find_sec(node.left), find_sec(node.right))
    x=find_sec(root)
    if x==float('inf'):
        return -1
    return x
```

## 674.最长连续递增序列

```python
def findLengthOfLCIS(self, nums: List[int]) -> int:
    ans = 0
    if len(nums) <= 1: return len(nums)

    tmp = [nums[0]]
    for i in range(1,len(nums)):
        if nums[i] <= nums[i-1]:
            if len(tmp) > ans:
                ans = len(tmp)
            tmp.clear()
        tmp.append(i)
    if len(tmp) > ans:
        ans = len(tmp)
    return ans
```

## 680.验证回文字符串II

```python
def validPalindrome(self, s: str) -> bool:
    slen = len(s)
    i, j= 0, slen-1
    while i<j:
        if s[i] == s[j]:
            i+=1
            j-=1
        else:
            break
    temp = s[i:j+1]
    a = temp[0:-1]
    if a == a[::-1]:
        return True
    b = temp[1:]
    if b == b[::-1]:
        return True      
    return False
```

## 682.棒球比赛

```python
def calPoints(self, ops: List[str]) -> int:
    ans = []
    for i in ops:
        if i =="C":
            ans.pop()
        elif i == "D":
            ans.append(ans[-1]*2)
        elif i == "+":
            ans.append(ans[-1] + ans[-2])
        else:
            ans.append(int(i))
    return sum(ans)
```

## 693.交替位二进制数

```python
def hasAlternatingBits(self, n: int) -> bool:
    return bin(n^(n>>1)).count('0') == 1
```

## 696.计数二进制子串

$$
我们可以将字符串 ss 按照 00 和 11 的连续段分组，存在 \textit{counts}counts 数组中，例如 s = 00111011s=00111011，可以得到这样的 \textit{counts}counts 数组：\textit{counts} = \{2, 3, 1, 2\}counts={2,3,1,2}。

这里 \textit{counts}counts 数组中两个相邻的数一定代表的是两种不同的字符。假设 \textit{counts}counts 数组中两个相邻的数字为 uu 或者 vv，它们对应着 uu 个 00 和 vv 个 11，或者 uu 个 11 和 vv 个 00。它们能组成的满足条件的子串数目为 \min \{ u, v \}min{u,v}，即一对相邻的数字对答案的贡献。
$$

```python
def countBinarySubstrings(self, s: str) -> int:
    seq = [0, 1]
    res = []
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            seq[1] += 1
        else:
            res.append(min(seq))
            seq[0] = seq[1]
            seq[1] = 1
    res.append(min(seq))
    return sum(res)
```

## 697.数组的度

```python
'''
在Python中如果访问字典中不存在的键，会引发KeyError异常（JavaScript中如果对象中不存在某个属性，则返回undefined）。但是有时候，字典中的每个键都存在默认值是非常方便的。通过 dict.setdefault() 方法来设置字典默认值。
dict.setdefault()方法接收两个参数，第一个参数是健的名称，第二个参数是默认值
'''
def findShortestSubArray(self, nums: List[int]) -> int:
    num_map = collections.defaultdict(list)
    for idx,num in enumerate(nums):
        num_map[num].append(idx)
    idx_lst = sorted(num_map.items(),key = lambda x: (len(x[1]),-1*(x[1][-1]-x[1][0])),reverse=True)[0][-1]
    return idx_lst[-1] - idx_lst[0] + 1
```

## 700.二叉搜索树中的搜索

```python
def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    if not root:
        return 0
    dq,depth = deque([root]), 1
    while dq:
        for i in range(len(dq)):
            tmp = dq.popleft()
            if tmp.val == val:
                return tmp
            if tmp.left:
                dq.append(tmp.left)
            if tmp.right:
                dq.append(tmp.right)
        depth += 1
    return None
```

## 703.数据流中的第K大元素

```python
'''
heapq 堆
堆是非线性的树形的数据结构，有两种堆,最大堆与最小堆。（ heapq库中的堆默认是最小堆）
最大堆，树种各个父节点的值总是大于或等于任何一个子节点的值。
最小堆，树种各个父节点的值总是小于或等于任何一个子节点的值。

heapq.heappush(heap, item)   heap为定义堆，item增加的元素

heapq.heapify(list)   将列表转换为堆

heapq.heappop(heap)  删除并返回最小值，因为堆的特征是heap[0]永远是最小的元素，所以一般都是删除第一个元素

heapq.heappushpop(list, item)
判断添加元素值与堆的第一个元素值对比；如果大，则删除并返回第一个元素，然后添加新元素值item.  如果小，则返回item. 原堆不变。

heapq.merge（） 将多个堆合并

heapq.nlargest(n,heap)  查询堆中的最大n个元素

heapq.nsmallest(n,heap)  查询堆中的最小n个元素


'''
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.que = nums
        heapq.heapify(self.que)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        heapq.heappush(self.que, val)
        while len(self.que) > self.k:
            heapq.heappop(self.que)
        return self.que[0]
```

## 704.二分查找

```python
def search(self, nums: List[int], target: int) -> int:
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid]  < target:
            l = mid + 1
        else:
            r = mid
    return -1
```

## 705.设计哈希集合

```python
class MyHashSet:
    def __init__(self):
        self.mod = 1007
        self.table = [[] for _ in range(self.mod)]

    def hash(self, key):
        return key % self.mod

    def div(self, key):
        return key // self.mod

    def add(self, key):
        hash_key = self.hash(key)
        if not self.table[hash_key]:
            self.table[hash_key] = [0] * self.mod
        self.table[hash_key][self.div(key)] = 1

    def remove(self, key):
        hash_key = self.hash(key)
        if self.table[hash_key]:
            self.table[hash_key][self.div(key)] = 0

    def contains(self, key):
        hash_key = self.hash(key)
        return self.table[hash_key] != [] and self.table[hash_key][self.div(key)] == 1
```

## 706.设计哈希映射

```python
class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.try_dict = {}


    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        self.try_dict[key] = value


    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        if key in self.try_dict:
            return self.try_dict[key]
        else:
            return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        if key in self.try_dict:
            del self.try_dict[key]
```

## 709.转换成小写字母

```python
def toLowerCase(self, s: str) -> str:
    return s.lower()
```

## 717.1比特与2比特字符

```python
def isOneBitCharacter(self, bits: List[int]) -> bool:
    bits = bits[::-1]
    while len(bits) > 1:
        if bits[-1] == 1:
            bits.pop()
        bits.pop()
    if len(bits) == 0:
        return False
    return bits[0] == 0
```

## 720.词典中最长的单词

```python
def longestWord(self, words: List[str]) -> str:
    words.sort()
    words_set, longest_word = set(['']), ''
    for word in words:
        if word[:-1] in words_set:
            words_set.add(word)
            if len(word) > len(longest_word):
                longest_word = word
    return longest_word
```

## 724.寻找数组的中心下标

```python
def pivotIndex(self, nums: List[int]) -> int:
    l_sum = 0
    r_sum = sum(nums)
    
    nums = [0] + nums[:]
    for i in range(1,len(nums)):
        r_sum -= nums[i]
        if l_sum == r_sum:
            return i-1
        else:
            l_sum += nums[i]
    return -1
```

## 728.自除数

```python
def selfDividingNumbers(self, left: int, right: int) -> List[int]:
    def isSelf(n):
        tn = n
        while n:
            t = n % 10
            if not t or tn % t:
                    return  False
            n = (n-t)//10
        return True
    ans = []
    for i in range(left,right+1):
        if isSelf(i):
            ans.append(i)
    return ans
```

## 733.图像渲染

```python
'''
在 Python 中，可以使用以下几种方法实现队列
collections包里的deque，对应操作
pop()从尾取出
appendleft() 从头插入
queue包中的queue，对应操作
put() 插入
get() 取出
直接使用list，只要保证只使用
pop() 取出
insert(0,) 插入
或者只使用
append() 插入
list[0]并且del list[0] 取出
两者使用list方法的不同就区别于你把哪个当头，哪个当尾
'''

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if newColor == image[sr][sc]:return image
        que, old,  = [(sr, sc)], image[sr][sc]
        while que:
            point = que.pop()
            image[point[0]][point[1]] = newColor
            for new_i, new_j in zip((point[0], point[0], point[0] + 1, point[0] - 1), (point[1] + 1, point[1] - 1, point[1], point[1])): 
                if 0 <= new_i < len(image) and 0 <= new_j < len(image[0]) and image[new_i][new_j] == old:  
                    que.insert(0,(new_i,new_j))
        return image
```

## 744.寻找比目标字母大的最小字母

```python
'''
字符转ascii数字 ord（char）
数字转字符  chr（int）
'''
def nextGreatestLetter(self, letters: List[str], target: str) -> str:
    t = ord(target)
    for i in letters:
        if  ord(i) >t:
            return i
    return letters[0]
```

## 746.使用最小花费爬楼梯

```python
'''
到达第i级台阶的阶梯顶部的最小花费，有两个选择：

先付出最小总花费minCost[i-1]到达第i级台阶（即第i-1级台阶的阶梯顶部），踏上第i级台阶需要再花费cost[i]，再迈一步到达第i级台阶的阶梯顶部，最小总花费为minCost[i-1] + cost[i])；

先付出最小总花费minCost[i-2]到达第i-1级台阶（即第i-2级台阶的阶梯顶部），踏上第i-1级台阶需要再花费cost[i-1]，再迈两步跨过第i级台阶直接到达第i级台阶的阶梯顶部，最小总花费为minCost[i-2] + cost[i-1])；

则minCost[i]是上面这两个最小总花费中的最小值。

minCost[i] = min(minCost[i-1] + cost[i], minCost[i-2] + cost[i-1])
'''
#解法1
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        minCost = [0] * n
        minCost[1] = min(cost[0], cost[1])
        for i in range(2, n):
            minCost[i] = min(minCost[i - 1] + cost[i], minCost[i - 2] + cost[i - 1])
        return minCost[-1]
    
 #解法2
'''
到达第i级台阶的阶梯顶部的最小花费，有两个选择：

最后踏上了第i级台阶，最小花费dp[i]，再迈一步到达第i级台阶楼层顶部；
最后踏上了第i-1级台阶，最小花费dp[i-1]，再迈两步跨过第i级台阶直接到达第i级台阶的阶梯顶部。
所以到达第i级台阶的阶梯顶部的最小花费为minCost[i] = min(dp[i], dp[i-1])。

即为了求出到达第i级台阶的阶梯顶部的最小花费，我们先算出踏上第i级台阶的最小花费，用dp[i]表示，再通过min(dp[i], dp[i-1])来求出到达第i级台阶的阶梯顶部的最小花费
'''
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0] * n
        dp[0], dp[1] = cost[0], cost[1]
        for i in range(2, n):
            dp[i] = min(dp[i - 2], dp[i - 1]) + cost[i]
        return min(dp[-2], dp[-1])
```

## 747.至少是其他数字两倍的最大数

```python
def dominantIndex(self, nums: List[int]) -> int:
    if len(nums) < 2: return -1
    tp = nums[:]
    nums.sort(reverse = True)
    if nums[0] >= nums[1]*2:
        return  tp.index(nums[0])
    else:
        return -1
```

## 748.最短补全词

```python
def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
    def counter(word):
        cnts = [0] * 26
        for c in word:
            if 'a' <= c <= 'z':
                cnts[ord(c) - ord('a')]+=1
            elif 'A' <= c <= 'Z':
                cnts[ord(c) - ord('A')]+=1
        return cnts
    
    cs = counter(licensePlate)
    l, ans = inf, None
    for w in words:
        if len(w) < l:
            ws, i = counter(w), 0
            while i < len(cs):
                if ws[i] < cs[i]:
                    break
                i += 1
            if i == len(cs):
                l, ans = len(w), w
    return ans
```

## 762.二进制表示中质数个计算置位

```python
#解法1
def countPrimeSetBits(self, left: int, right: int) -> int:
    #判断是否为质数函数
    def isPrime(x):
        for j in range(2, x//2 + 1):
            if not x % j:
                return False
        return x > 1
    
    ans = 0
    for i in range(left,right+1):
        c = bin(i).count('1')
        if isPrime(c):
            ans += 1
    return ans

#解法2  用bit_count()函数
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19}
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        return sum(i.bit_count() in PRIMES for i in range(left, right + 1))
```

## 766.托普利兹矩阵

```python
#只要每行的除了最后一位的所有元素与下面一行的除了第一位的所有元素相同即可（即保证每个元素与自己的右下角元素相同）
def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
    for i in range(len(matrix) - 1):
        if matrix[i][:-1] != matrix[i + 1][1:]:
            return False
    return True
```

## 771.宝石与石头

```python
def numJewelsInStones(self, jewels: str, stones: str) -> int:
    ans = 0
    for i in stones:
        if i in jewels:
            ans += 1
    return ans
```

## 783.二叉搜索树节点的最小距离

```
def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    ret = []

    def dfs(root):
        if not root:
            return None
        dfs(root.left)
        ret.append(root.val)
        dfs(root.right)

    dfs(root)
    return min(ret[i+1] - ret[i] for i in range(len(ret) -1)) 
```

## 796.旋转字符串

```python
def rotateString(self, s: str, goal: str) -> bool:
    return len(s) == len(goal) and goal in s + s
```

## 804.唯一的摩尔斯密码词

```python
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        alphabt = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."][::-1]

        ans = []
        for i in words:
            tp = ""
            for ch in i:
                tp += alphabt[122 - ord(ch)]
            if tp not in ans:
                ans.append(tp)
        return len(ans)
```

## 806.写字符串需要的行数

```python
# ord（int）  单字符转数字
def numberOfLines(self, widths: List[int], s: str) -> List[int]:
    count = 1
    tp = 0
    widths = widths[::-1]
    for i in s:
        if tp + widths[122 - ord(i)] > 100:
            count += 1
            tp = widths[122 - ord(i)]
        else:
            tp += widths[122 - ord(i)]
    return [count, tp]
```

## 812.最大三角形面积

```python
#解法1  combinations函数
'''
itertools.combinations(iterable, r)
从可迭代对象iterable中选取r个单位进行组合，并返回一个生成元组的迭代器

from itertools import combinations
a = 'abc'   #对字符串进行combinations排列组合
for i in combinations(a,2):
    x = ''.join(i)
    print (x,end=' ')
输出：ab, ac, bc
'''
def largestTriangleArea(self, points: List[List[int]]) -> float:
    return max(abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2 for (x1, y1), (x2, y2), (x3, y3) in combinations(points, 3))


#解法2 海伦公式
  def largestTriangleArea(self, points: List[List[int]]) -> float:
        # 判断三个点是否可以组成三角形，同时更新三角形面积
        res = 0
        n = len(points)
        for i in range(n - 2):
            x = points[i]
            for j in range(i + 1, n - 1):
                y = points[j]
                for k in range(j + 1, n):
                    z = points[k]
                    if (x[1] - y[1]) * (z[0] - y[0]) == (z[1] - y[1]) * (x[0] - y[0]):
                        continue
                    else:
                        '''
                        # 海伦公式
                        S = √p(p-a)(p-b)(p-c)
                        
                        a = sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
                        b = sqrt((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2)
                        c = sqrt((y[0] - z[0]) ** 2 + (y[1] - z[1]) ** 2)
                        p = (a + b + c) / 2
                        rect_area = sqrt(p * (p - a) * (p - b) * (p - c))
                        '''
                        # 鞋带公式
                        a = x[0] * y[1] + y[0] * z[1] + z[0] * x[1]
                        b = x[1] * y[0] + y[1] * z[0] + z[1] * x[0]
                        rect_area = abs(a - b) / 2
                        res = max(res, rect_area)
        return res
```

## 819.最常见的单词

```python
'''
方法一：通过sorted()函数排序所有的value
import  operator
# 先通过sorted 和operator 函数对字典进行排序，然后输出value的键
classCount={"c":1,"b":4,"d":2,"e":6}
print(classCount.items())
SortedclassCount1= sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
print(SortedclassCount1[0][0])
 
# 通过max求字典value对应的key
print(max(classCount,key=classCount.get))


方法二：通过max函数取字典中value所对应的key值
# 例：
price = {
    'a':1,
    'b':7,
    'c':5,
    'd':10,
    'e':12，
    'f':3
}
result_max = max(price,key=lambda x:price[x])
print(f'max:{result_max}')
>>>>
max:e
'''

re模块称为正则表达式

#解法1  Counter函数
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        return max(Counter(re.split(r"[ ,.!?';]", paragraph.lower())).items(), key=lambda x:(len(x) > 0, x[0] not in b, x[1]))[0] if (b := set(banned + [""])) else ""
    
#解法2 
def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
    paragraph = re.sub(r'[^\w\s]', ' ', paragraph).lower() # 去除段落里的标点符号
    paragraph = paragraph.split()
    dicts = {}
    for char in paragraph: # 统计paragraph里单词出现的次数，这段代码也可以直接使用collections.Counter函数代替
        if char not in dicts: # dicts = collections.Counter(paragraph)
            dicts[char] = 1
        else:
            dicts[char] += 1
    for i in dicts: # j将在banned里的键所对应的值，置为-1，这样在比较大小时，可以将其忽略。
        if i in banned:
            dicts[i] = -1
    return max(dicts, key = dicts.get) # 获得字典dicts中value的最大值所对应的键
```

## 821.字符的最短距离

```python
#解法1双指针
def shortestToChar(self, s: str, c: str) -> List[int]:
    ans, last = [inf] * len(s), None
    for i, ch in enumerate(s):
        if ch == c:
            if last is not None:
                for j in range(i, (i - 1 + last) // 2 - 1, -1):
                    ans[j] = min(ans[j], i - j)
            else:
                for j in range(i, -1, -1):
                    ans[j] = min(ans[j], i - j)
            last = i
        elif last is not None:
            ans[i] = min(ans[i], i - last)
    return ans
```

## 824.山羊拉丁文

```python
def toGoatLatin(self, sentence: str) -> str:
    vowel = ['a', 'e', 'i', 'o', 'u','A','E','I','O','U']
    sp = sentence.split()
    for i in range(len(sp)):
        if sp[i][0] not in vowel:
            sp[i] = sp[i][1:] + sp[i][0]
        sp[i] += 'ma' + 'a'*(i+1)
    return ' '.join(sp)
```

## 830.较大分组的位置

```python
#双指针
def largeGroupPositions(self, s: str) -> List[List[int]]:
    l,r =0, 0
    ans = []
    while r < len(s):
        if s[r] != s[l]:
            if r-l > 2:
                ans.append([l, r-1])
            l = r
        r += 1
    if r-l > 2:
        ans.append([l, r-1])
    return ans
```

## 832.翻转图像

```python
#利用lambda表达式批量修改数组元素
def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
    ans = []
    for i in image:
        ans.append(list(map(lambda x:1-x, i[::-1])))
    return ans
```

## 836.矩形重叠

```python
def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
    return not(rec1[2] <= rec2[0] or rec2[2] <= rec1[0]) and not(rec1[3] <= rec2[1] or rec2[3] <= rec1[1])
```

## 844.比较含退格的字符串

```python
def backspaceCompare(self, s: str, t: str) -> bool:
    s1 = []
    s2 = []
    for i in s:
        if i == "#":
            if len(s1):
                s1.pop()
        else:
            s1.append(i)
    for j in t:
        if j == "#":
            if len(s2):
                s2.pop()
        else:
            s2.append(j) 
    return s1 == s2
```

## 859.亲密字符串

```python
def buddyStrings(self, s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
   	#判断是否有重复元素
    if s == goal and len(set(s)) < len(s):
        return True
    diff = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            diff.append(i)
    if len(diff) == 2:
        return True if s[diff[0]] == goal[diff[1]] and s[diff[1]] == goal[diff[0]] else False
    else:
        return False
```

## 860.柠檬水找零

```python
def lemonadeChange(self, bills: List[int]) -> bool:
    s5 = 0
    s10 = 0

    for i in bills:
        if i == 5:
            s5 += 1
        elif i == 10:
            if s5 > 0:
                s5 -= 1
                s10 += 1
            else:
                return False
        else:
            if s10 > 0 and s5 > 0:
                s10 -= 1
                s5 -= 1
            elif s5 >= 3:
                s5 -= 3
            else:
                return False
    return True
```

## 867.转置矩阵

```python
#解法1   k = list(zip(*matrix)) #注意zip返回的是tuple不是list

def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
    k = list(zip(*matrix))
    tmp = []
    for i in k:
        tmp.append(list(i))
    return tmp

#解法2
def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
```

## 869.二进制间距

```python
def binaryGap(self, n: int) -> int:
    ans = 0
    st = str(bin(n))[2:]
    l,r = 0, 0
    for i in range(len(st)):
        if st[i] == '1':
            r = i
            if r - l > ans:
                ans = r - l
            l = r
    return ans
```

## 872.叶子相似的树

```python
def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    que = [root1]
    que2 = [root2]
    ans1 = []
    ans2 = []
    while len(que):
        i = que.pop()
        if i:
            if not i.left and not i.right:
                ans1.append(i.val)
            elif i.left or i.right:
                que.append(i.left)
                que.append(i.right)
    while len(que2):
        j = que2.pop()
        if j:
            if not j.left and not j.right:
                ans2.append(j.val)
            elif j.left or j.right:
                que2.append(j.left)
                que2.append(j.right)
    return ans1 == ans2
```

## 876.链表的中间结点

```python
def middleNode(self, head: ListNode) -> ListNode:
    ls = []
    while head:
        ls.append(head)
        head = head.next
    return ls[len(ls)//2]
```

## 883.三维形体投影面积

```python
def projectionArea(self, grid: List[List[int]]) -> int:
    n = len(grid)
    x = sum([max(row) for row in grid])
    y = sum([max([grid[i][j] for i in range(n)]) for j in range(n)])
    z = sum([sum([1 if grid[i][j] != 0 else 0 for i in range(n)]) for j in range(n)])
    return x + y + z
```

## 884.两句话中的不常见单词

```python
'''
数组删除元素
1.remove: 删除单个元素，删除首个符合条件的元素，按值删除
举例说明:
>>> str=[1,2,3,4,5,2,6]
>>> str.remove(2)
>>> str
[1, 3, 4, 5, 2, 6]


2.pop:  删除单个或多个元素，按位删除(根据索引删除)
>>> str=[0,1,2,3,4,5,6]
>>> str.pop(1)   #pop删除时会返回被删除的元素
>>> str
[0, 2, 3, 4, 5, 6]
>>> str2=['abc','bcd','dce']
>>> str2.pop(2)
'dce'
>>> str2
['abc', 'bcd']


3.del：它是根据索引(元素所在位置)来删除
举例说明:
>>> str=[1,2,3,4,5,2,6]
>>> del str[1]
>>> str
[1, 3, 4, 5, 2, 6]
>>> str2=['abc','bcd','dce']
>>> del str2[1]
>>> str2
['abc', 'dce']


除此之外，del还可以删除指定范围内的值。
>>> str=[0,1,2,3,4,5,6]
>>> del str[2:4]  #删除从第2个元素开始，到第4个为止的元素(但是不包括尾部元素)
>>> str
[0, 1, 4, 5, 6]
'''

#用counter函数
def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        return [key for key,val in collections.Counter((s1+' '+s2).split()).items()  
                if val == 1 ]

def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
    word, ans = Counter(s1.split()+s2.split()), []
    for w, i in filter(lambda x: x[1] == 1, word.items()):
        ans.append(w)
    return ans
```

## 888.公平的糖果交换

```python
def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
    suma = sum(aliceSizes)
    sumb = sum(bobSizes)
    diff = (sumb - suma)//2
    for i in aliceSizes:
        if i+diff in bobSizes:
            return [i, i+diff]
    return []
```

## 892.三维形体的表面积

```python
def surfaceArea(self, grid: List[List[int]]) -> int:
    count = 0 
    minus = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if i - 1 >= 0:
                minus += min(grid[i][j] , grid[i-1][j])
            if i + 1 < len(grid):
                minus += min(grid[i+1][j], grid[i][j])
            if j - 1 >= 0:
                minus += min(grid[i][j] , grid[i][j-1])
            if j + 1 < len(grid[i]):
                minus += min(grid[i][j] , grid[i][j+1])
            if grid[i][j] > 0:
                count += grid[i][j]
                minus += 2*(grid[i][j] -1)
    return count*6 - minus
```

## 896.单调数列

```python
def isMonotonic(self, nums: List[int]) -> bool:
    t = nums[:]
    nums.sort()
    return t == nums or t == nums[::-1]
```

## 897.递增顺序搜索树

```python
#树的中序遍历  「二叉搜索树（BST）的中序遍历是有序的」
'''
先序遍历：
def dfs(root):
    if not root:
        return
    执行操作
    dfs(root.left)
    dfs(root.right)
中序遍历：
def dfs(root):
    if not root:
        return
    dfs(root.left)
    执行操作
    dfs(root.right)
后序遍历：
def dfs(root):
    if not root:
        return
    dfs(root.left)
    dfs(root.right)
	执行操作
'''
def increasingBST(self, root: TreeNode) -> TreeNode:
    def DFS(root):
        if not root: return
        DFS(root.left)
        self.newtree.right = TreeNode(root.val)
        self.newtree = self.newtree.right
        DFS(root.right)
    newTreeHEAD = TreeNode()
    self.newtree = newTreeHEAD
    DFS(root)
    return newTreeHEAD.right
```

## 905.按奇偶排序数组

```python
def sortArrayByParity(self, nums: List[int]) -> List[int]:
    t = Counter(nums)
    odd = 0
    even = len(nums)-1

    while odd < even:
        if not nums[odd] % 2:
            odd += 1
        if nums[even] % 2:
            even -= 1
        elif nums[odd] % 2 and not nums[even] % 2:
            nums[odd], nums[even] = nums[even], nums[odd]
    
    return nums
```

## 908.最小差值I

```python
def smallestRangeI(self, nums: List[int], k: int) -> int:
    nums.sort()
    mi = nums[0]
    mx = nums[-1]
    if mi + k >= mx -k:
        return 0
    return mx - mi - 2*k
```

## 914.卡牌分组

```python
#解法1 reduce + gcd
'''
reduce函数先从列表（或序列）中取出2个元素执行指定函数，并将输出结果与第3个元素传入函数，输出结果再与第4个元素传入函数，…，以此类推，直到列表每个元素都取完。

math.gcd(*integers)  返回多个整数的最大公约数 
'''
def hasGroupsSizeX(self, deck: List[int]) -> bool:
    return reduce(gcd, Counter(deck).values()) >= 2


#解法2 遍历求最大公约数
def hasGroupsSizeX(self, deck: List[int]) -> bool:
    counter = sorted(list(set(dict(Counter(deck)).values())))
    min = counter[0]
    if min == 1:
        return False
    for i in range(2, min+1):
        if all(c % i == 0 for c in counter):
            return True
    return False
```

## 917.仅仅反转字母

```python
def reverseOnlyLetters(self, s: str) -> str:
    t = list(s)
    l,r =0, len(t)-1
    while l < r:
        if not t[l].isalpha():
            l += 1
        if not t[r].isalpha():
            r -= 1
        if t[l].isalpha() and t[r].isalpha():
            t[l], t[r] = t[r], t[l]
            l += 1
            r -= 1
    return ''.join(t)
```

## 922.按奇偶排序数组II

```python
def sortArrayByParityII(self, nums: List[int]) -> List[int]:
    l,r = 0, 0
    t = len(nums)
    while l < t and r < t:
        if not nums[l]%2 and l%2 and nums[r]%2 and not r%2:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r += 1
        else:
            if nums[l] % 2 or not l % 2:
                l += 1
            if not nums[r] % 2 or r % 2:
                r += 1
    return nums
```

## 925.长按键入

```python
#双指针
def isLongPressedName(self, name: str, typed: str) -> bool:
    i = j = 0
    while j < len(typed):
        if i < len(name) and typed[j] == name[i]:
            i += 1
            j += 1
        elif j > 0 and typed[j] == typed[j - 1]:
            j += 1
        else:
            return False
    return i == len(name)
```

## 929.独特的电子邮件地址

```python
def numUniqueEmails(self, emails: List[str]) -> int:
    ans = []
    for i in emails:
        st = i[:i.index('@')]
        tmp = st.replace(".", "")
        if tmp.find('+') != -1:
            tmp = tmp[:tmp.find('+')]
        tmp += i[i.index('@'):]
        if tmp not in ans:
            ans.append(tmp)
    return len(ans)
```

## 933.最近的请求次数

```python
class RecentCounter:
    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        while self.queue and self.queue[0] < t - 3000:
            self.queue.popleft()
        self.queue.append(t)
        return len(self.queue)
```

## 937.重新排列日志文件

```python
'''
自定义排序
 def takeSecond(elem):
            return elem.split()[1:]
        alp.sort(key = takeSecond)
'''
def reorderLogFiles(self, logs: List[str]) -> List[str]:
    # 自定义排序器，返回一个key元组用于排序
    def comp(log: str) -> tuple:
        # a为标识符，b为内容
        a, b = log.split(' ', maxsplit=1)
        if b[0].isalpha(): # 如果是字母日志
            return (0, b, a) # 先按内容字母排序，相同则按标识符排序
        else:
            return (1,) # 1是数字日志，确保数字日志排在字母日志之后

    logs.sort(key=comp)  # sort 是稳定排序
    return logs
```

## 938.二叉搜索树的范围和

```python
def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
    res = []
    def dfs(tree, res):
        if tree == None:
            return
        dfs(tree.left, res)
        if tree.val >= low and tree.val <= high:
            res.append(tree.val)
        dfs(tree.right, res)
    dfs(root, res)
    return sum(res)
```

## 941.有效的山脉数组

```python
#双指针
def validMountainArray(self, A: List[int]) -> bool:
    n = len(A)
    left = 0
    right = n - 1

    while left + 1 < n and A[left] < A[left+1]:
        left += 1
    while right - 1 > 0 and A[right - 1] > A[right]:
        right -= 1
    # 判断 left 指针是否与 right 指针重合
    # 同时注意，峰顶不能在数组两端
    if left > 0 and right < n - 1 and left == right:
        return True
    return False
```

## 942.递增字符串匹配

```python
def diStringMatch(self, s: str) -> List[int]:
    left, right, ans = 0, len(s), []
    for c in s:
        if c == 'I':
            ans.append(left)
            left += 1
        else:
            ans.append(right)
            right -= 1
    ans.append(left)
    return ans
```

## 944.删列造序

```python
#运用zip函数
def minDeletionSize(self, strs: List[str]) -> int:
    return sum(list(column) != sorted(column) for column in zip(*strs))
```

## 953.验证外星语词典

```python
def isAlienSorted(self, words: List[str], order: str) -> bool:
    ord_map = {j: i for i, j in enumerate(order)}
    pre = ""
    for word in words:
        # 比较 pre 和 word
        print(list(zip(pre, word)))
        for i, j in zip(pre, word):
            if i != j:
                if ord_map[i] > ord_map[j]:
                    return False
                else:
                    break
        else:
            if len(pre) > len(word):
                return False
        pre = word
    return True
```

## 961.在长度2N的数组中找出重复N次的元素

```python
def repeatedNTimes(self, nums: List[int]) -> int:
    s = set()
    for i in nums:
        if i in s:
            return i
        else:
            s.add(i)
    return -1
```

## 965.单值二叉树

```python
def isUnivalTree(self, root: TreeNode) -> bool:
    tp = -1
    def dfs(tree,tp):
        if tree == None:
            return True
        if tp == -1:
            tp = tree.val
        else:
            if tree.val != tp:
                return False
        return dfs(tree.left,tp) and dfs(tree.right,tp)
    return dfs(root, tp)
```

## 976.三角形的最大周长

```python
#贪心法
def largestPerimeter(self, nums: List[int]) -> int:
    def isTri(a: int, b: int, c: int) -> bool:
        return a < b + c

    nums.sort(reverse=True)
    for i in range(2, len(nums)):
        if isTri(nums[i-2], nums[i-1], nums[i]):
            return nums[i-2] + nums[i-1] + nums[i]
    return 0
```

## 977.有序数组的平方

```python
def sortedSquares(self, nums: List[int]) -> List[int]:
    for i in range(len(nums)):
        nums[i] = nums[i]**2
    nums.sort()
    return nums
```

## 989.数组形成的整数加法

```python
#模拟加法进位
def addToArrayForm(self, A: List[int], K: int) -> List[int]:
    i = len(A) - 1
    while K:
        A[i] += K
        K, A[i] = A[i] // 10, A[i] % 10
        i -= 1

        if i < 0 and K:
            A.insert(0,0)
            i = 0
    return A
```

## 993.二叉树的堂兄弟节点

```python
def isCousins(self, root, x, y):
    result_x = []
    result_y = []
    def dfs(node,parent,deep):
        if not node:
            return
        nonlocal result_x, result_y
        if node.val == x:
            result_x = [parent,deep]
        elif node.val == y:
            result_y = [parent,deep]
        dfs(node.left,node,deep + 1)
        dfs(node.right,node,deep + 1)
    dfs(root,root,0)
    return result_x[0] != result_y[0] and result_x[1] == result_y[1]
```

## 997.找到小镇的法官

```python
def findJudge(self, n: int, trust: List[List[int]]) -> int:
    s, l = set(), [0] * n
    for a, b in trust:
        s.add(a)
        l[b-1] += 1
    for i in range(1, n + 1):
        if i not in s and l[i - 1] == n - 1:
            return i
    return -1
```

## 999.可以被一步捕获的棋子数

```python
def numRookCaptures(self, board):
    return (lambda x:x('pR')+x('Rp'))([''.join(board[x]+[' ']+[i[y] for i in board]).replace('.','') for x in range(8) for y in range(8) if board[x][y]=='R'][0].count)
```

## 1002.查找共用字符

```python
#解法1 字典
def commonChars(self, words: List[str]) -> List[str]:
    st = ""
    l = len(words)
    ans = Counter(words[0])

    if l <= 1:
        return words
    else:
        for i in words[1:]:
            ct = Counter(i)
            for ele in ans.keys():
                if not ele in i:
                    ans[ele] = 0
            for ele,c in ct.items():
                if ele in ans.keys():
                    ans[ele] = min(ans[ele], c)

        res = []

        for ele, c in ans.items():
            while c:
                res.append(ele)
                c -= 1
        return res
    
#解法2
def commonChars(self, A: List[str]) -> List[str]:
    res = []
    min_length_char = min(A, key=len)
    for char in min_length_char:
        if all(char in item  for item in A):
            res.append(char)
            A = [i.replace(char,'',1)  for i in A]

    return res
```

## 1005.K次取反后最大化的数组和

```python
#解法1 每次翻转最后的然后排序
def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
    nums.sort(reverse = True)
    while k:
        k -= 1
        nums[-1] = -nums[-1]
        nums.sort(reverse = True)
    return sum(nums)


#解法2  贪心法 ，在sort函数中加入key = abs实现不看符号的大小排序
def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
    nums.sort(key = abs, reverse = True)    #排序时加入key = abs 
    for i in range(len(nums)):
        if nums[i] < 0 and k > 0:
            nums[i] *= -1
            k -= 1
    if k % 2 == 1: nums[-1] *= -1
    return sum(nums)
```

## 1009.十进制整数的反码

```python
#解法1 求出最高位
def bitwiseComplement(self, n: int) -> int:
    def bitwiseComplement(self, n: int) -> int:
    ct = len(bin(n))-2
    return 2**(ct) - n - 1

#解法2 字符串替换
def bitwiseComplement(self, N: int) -> int:
    x=str(bin(N))[2:].replace('1','2').replace('0','1').replace('2','0')
    return int(x,2)
```

## 1013.将数组分成和相等的三个部分

```python
#解法1
def canThreePartsEqualSum(self, arr: List[int]) -> bool:
    sm = sum(arr)//3
    tp = 0
    tar = 0
    for i in range(len(arr)):
        tp += arr[i]
        if tp == sm:
            tp = 0
            tar += 1
            if tar == 2:
                tp = i
                if i == len(arr) - 1:
                    return False
                break
    return sum(arr[tp+1:]) == sm

#解法2
def canThreePartsEqualSum(self, A: List[int]) -> bool:
    sum = 0
    for x in A:
        sum += x
    # 和不能被3整除，肯定不符合
    if sum % 3:
        return False
    
    left, right = 0, len(A)-1
    leftSum, rightSum = A[left], A[right]

    # left + 1 < right: 防止将数组只分成两部分，中间部分至少要有一个元素
    while left + 1 < right:
        # 左右都等于sum/3，中间肯定等于sum/3
        if leftSum == sum/3 and rightSum == sum/3:
            return True
        if leftSum != sum/3:
            left += 1
            leftSum += A[left]
        if rightSum != sum/3:
            right -= 1
            rightSum += A[right]
    return False
```

## 1018.可被5整除的二进制前缀

```python
def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
    rec = 0
    ans = []
    for i in nums:
        if i == 1:
            rec = rec*2 + 1
        else:
            rec *= 2
        if rec % 5:
            ans.append(False)
        else:
            ans.append(True)
    return ans 
```

## 1021.删除最外层的括号

```python
def removeOuterParentheses(self, s: str) -> str:
    i = j = cur = 0
    ans = ""
    while i < len(s):
        cur += 1
        j += 1
        while j < len(s) and cur:
            cur += 1 if s[j] == '(' else -1
            j += 1
        ans += s[i + 1:j - 1]
        i = j
    return ans
```

## 1022.从根到叶的二进制之和

```python
def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
    def dfs(tree, res):
        if tree == None:
                return 0
        res = res*2 + tree.val
        if tree.left == None and tree.right == None:
            print(res)
            return res
        return dfs(tree.left, res) + dfs(tree.right, res)
    return dfs(root, 0)
```

## 1025.除数博弈

```python
#博弈论
def divisorGame(self, n: int) -> bool:
    if n%2==0:
        return True
    else:
        return False
```

## 1030.距离顺序排列矩阵单元格

```python
#运用key和lambda
def allCellsDistOrder(self, R: int, C: int, r0: int, c0: int) -> List[List[int]]:
    return sorted([[i,j] for i in range(R) for j in range(C)],
            key = lambda x : abs((x[0]-r0)) + abs((x[1]-c0)))
```

## 1037.有效的回旋镖

```python
def isBoomerang(self, points: List[List[int]]) -> bool:
    return (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) != (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
```

## 1046.最后一块石头的重要

```python
'''
python  heapq只有小顶堆，可以采用*-1来实现
通常调用Python中的heapq包来实现堆Heap：

heapify():将列表List转变为堆Heap（默认转变为小堆）

heappush( , ):添加一个值在堆中

heappop():删除一个值在堆中

nlargest():前n个最大值

nsmallest():前n个最小值
'''
def lastStoneWeight(self, stones: List[int]) -> int:
    # 初始化
    heap = [-stone for stone in stones]
    heapq.heapify(heap)

    # 模拟
    while len(heap) > 1:
        x,y = heapq.heappop(heap),heapq.heappop(heap)
        if x != y:
            heapq.heappush(heap,x-y)

    if heap: return -heap[0]
    return 0
```

## 1047.删除字符串中的所有相邻重复项

```python
def removeDuplicates(self, s: str) -> str:
    stack = []
    for i in s:
        if not stack:
            stack.append(i)
            continue
        if stack and stack[-1] != i:
            stack.append(i)
        elif stack and stack[-1] == i:
            stack.pop()
    return ''.join(stack)
```

## 1051.高度检查器

```python
def heightChecker(self, heights: List[int]) -> int:
    tp = heights[:]
    heights.sort()
    ans  = 0
    for i in range(len(tp)):
        if tp[i] != heights[i]:
            ans += 1
    return ans 
#简写版  return sum(a != b for a, b in zip(heights, sorted(heights)))
```

## 1071.字符串的最大因子

```python
def gcdOfStrings(self, str1: str, str2: str) -> str:
    def GCD(a: int, b: int) -> int:
        return a if b == 0 else GCD(b, a % b)
    m,n = len(str1),len(str2)
    l = GCD(m,n)
    dm,dn = m//l,n//l
    tem =  str1[0:l]
    return tem if str1==tem*dm and str2 == tem*dn else ""
```

## 1078.Bigram分词

```python
def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
    st = text.split()
    ans = []
    for i in range(len(st)-2):
        if st[i] == first and st[i+1] == second:
            ans.append(st[i+2])
    return ans
```

## 1089.复写零

```python
def duplicateZeros(self, arr: List[int]) -> None:
    """
    Do not return anything, modify arr in-place instead.
    """
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            arr.insert(i+1, 0)
            arr.pop()
            i += 1
        i += 1
```

## 1103.分糖果

```python
def distributeCandies(self, candies: int, num_people: int) -> List[int]:
    ans  = [0]*num_people
    index  = 1
    pos = 0
    while candies >= index:
            candies -= index
            ans[pos] += index
            index += 1 
            pos += 1
            if pos == num_people:
                pos = 0
    ans[pos] += candies
    return ans
```

## 1108.IP无效化

```python
def defangIPaddr(self, address: str) -> str:
    return address.replace('.',"[.]")
```

##  1114.按序打印

```python
#方法1 Lock方法
import threading
class Foo:
    def __init__(self):
        self.l1 = threading.Lock()
        self.l1.acquire()
        self.l2 = threading.Lock()
        self.l2.acquire()

    def first(self, printFirst: 'Callable[[], None]') -> None:
        printFirst()
        self.l1.release()

    def second(self, printSecond: 'Callable[[], None]') -> None:
        self.l1.acquire()
        printSecond()
        self.l2.release()

    def third(self, printThird: 'Callable[[], None]') -> None:
        self.l2.acquire()
        printThird()
        
        
#方法二，Condition条件对象法：
threading模块里的Condition方法，后面五种的方法也都是调用这个模块和使用不同的方法了，方法就是启动wait_for来阻塞每个函数，直到指示self.t为目标值的时候才释放线程，with是配合Condition方法常用的语法糖，主要是替代try语句的。

import threading
class Foo:
    def __init__(self):
        self.c = threading.Condition()
        self.t = 0

    def first(self, printFirst: 'Callable[[], None]') -> None:
        self.res(0, printFirst)

    def second(self, printSecond: 'Callable[[], None]') -> None:
        self.res(1, printSecond)

    def third(self, printThird: 'Callable[[], None]') -> None:
        self.res(2, printThird)
        
    def res(self, val: int, func: 'Callable[[], None]') -> None:
        with self.c:
            self.c.wait_for(lambda: val == self.t) #参数是函数对象，返回值是bool类型
            func()
            self.t += 1
            self.c.notify_all()  
```

## 1122.数组的相对排序

```python
def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    tp = []
    tp2 = []
    for i in arr1:
        if i in arr2:
            tp.append(i)
        else:
            tp2.append(i)
    return sorted(tp, key = lambda x : arr2.index(x)) + sorted(tp2)
```

## 1128.等价多米诺骨牌对的数量

```python
#运用collections.defaultdict
def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
    domino_map = collections.defaultdict(int)      
    for i,j in dominoes:
        num = 10 * i + j if i < j else 10 * j+ i
        domino_map[num] += 1
    return sum(n*(n-1)//2  for n in domino_map.values())
```

## 1137.第N个泰波那契

```python
 def tribonacci(self, n: int) -> int:
    tp = [0,1,1]
    i = 3
    while i <= n:
        tp.append(tp[i-1] + tp[i-2] + tp[i-3])
        i += 1
    return tp[n]
```

## 1154.一年中的第几天

```python
import calendar
class Solution:
    def dayOfYear(self, date: str) -> int:
        long = [31,28,31,30,31,30,31,31,30,31,30,31]
        y,m,d = int(date.split('-')[0]),int(date.split('-')[1]),int(date.split('-')[2])

        判断闰年的方法
        # if (year % 4) == 0:
        # if (year % 100) == 0:
        #     if (year % 400) == 0:
        #         print(f"{year}是闰年")
        #     else:
        #         print(f"{year}不是闰年")
        # else:
        #     print(f"{year}是闰年")
        if calendar.isleap(y):			#或者直接用库函数
            long[1] += 1
        return sum(long[:m-1]) + d
```

## 1160.拼写单词

```python
#方法1
import copy
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        ans = 0
        tp = Counter(chars)
        for i in words:
            t = copy.deepcopy(tp) #用copy来实现深拷贝
            len = 0
            for c in i:
                if c not in t.keys() or not t[c]:
                    len = 0
                    break
                else: 
                    t[c] -= 1
                    len += 1
            ans += len
        return ans
    
#方法2  用all函数进行判断
def countCharacters(self, words: List[str], chars: str) -> int:
    ans = 0
    cnt = collections.Counter(chars)
    for w in words:
        c = collections.Counter(w)
        if all([c[i] <= cnt[i] for i in c]):
            ans += len(w)
    return ans
```

## 1175.质数排列

```python
 def numPrimeArrangements(self, n: int) -> int:
        mod = int(1e9+7)
        p, np = 0, 1

        #判断是否为质数
        def isprime(x):
            for i in range(2, x):
                if x % i == 0:
                    return False
                elif i * i > x:
                    return True
            return True
        
        for i in range(2, n + 1):
            if isprime(i):
                p += 1
            else:
                np += 1
        
        ans = 1
        for i in chain(range(1, p + 1), range(1, np + 1)): #chain函数用来对不同集合执行相同操作
            ans = (ans * i) % mod
        return ans
```

## 1184.公交站间的距离

```python
def distanceBetweenBusStops(self, distance: List[int], start: int, destination: int) -> int:
    sum1 = sum(distance)
    sum2 = 0
    tip1 = min(start,destination)#注意，由于循环模拟过程是从小到大的，如果遇到起点的编号大于终点，那么可以采取反转的方法（基于两个方向的路程是相同的）
    tip2 = max(start,destination)
    while tip1 < tip2:
        sum2 += distance[tip1]
        tip1 += 1
    return min(sum1-sum2,sum2)#由于环形路程有两种，那么需要选择其中最小的一种（基于环形路的基础，两种路程的和是列表数值的和）
```

## 1185.一周中的第几天

**蔡勒（Zeller）公式**，是一个计算星期的公式，随便给一个日期，就能用这个公式推算出是星期几:
$$
w =  y + [\frac{y}{4}] + [\frac{c}{4]}] - 2c + [\frac{13(m+1)}{5}] +d + 2
$$
**w**：星期； w对7取模得：0-星期日，1-星期一，2-星期二，3-星期三，4-星期四，5-星期五，6-星期六

**c**：世纪（注：一般情况下，在公式中取值为已经过的世纪数，也就是年份除以一百的结果，而非正在进行的世纪，也就是现在常用的年份除以一百加一；不过如果年份是公元前的年份且非整百数的话，c应该等于所在世纪的编号，如公元前253年，是公元前3世纪，c就等于-3）

**y**：年（一般情况下是后两位数，如果是公元前的年份且非整百数，y应该等于cMOD100+100）

**m**：月（m大于等于3，小于等于14，即在蔡勒公式中，某年的1、2月要看作上一年的13、14月来计算，比如2003年1月1日要看作2002年的13月1日来计算）

**d**：日

> 不过，蔡勒公式只适合于1582年（中国明朝万历十年）10月15日之后的情形。罗马教皇**格里高利十三世**在1582年组织了一批天文学家，根据[哥白尼日心说](https://baike.baidu.com/item/哥白尼日心说/4733578)计算出来的数据，对[儒略历](https://baike.baidu.com/item/儒略历)作了修改。将**1582年10月5日到14日**之间的10天宣布撤销，继10月4日之后为10月15日。后来人们将这一新的历法称为“[格里高利历](https://baike.baidu.com/item/格里高利历/9556174)”，也就是今天世界上所通用的历法，简称[格里历](https://baike.baidu.com/item/格里历)或[公历](https://baike.baidu.com/item/公历)。

[ ]代表取整，即只要整数部分。

```python
#解法1 计算距离今天的差值
ANS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        def helper(d, m, y):
            # 2022.1.3 星期一
            ans = 1 
            # 天数的偏移量
            ans = (ans + d - 3) % 7
            # 计算年的偏差量
            if y < 2022:
                for i in range(y, 2022):
                    ans = (ans - (366 if not i % 4 and (i % 100 or not i % 400) else 365)) % 7
            else:
                for i in range(2022, y):
                    ans = (ans + (366 if not i % 4 and (i % 100 or not i % 400) else 365)) % 7
            # 计算月的偏差量
            for i in range(m - 1):
                ans = (ans + DAYS[i]) % 7
                if i == 1 and not y % 4 and (y % 100 or not y % 400):
                    ans = (ans + 1) % 7
            return ans
        return ANS[helper(day, month, year)]
    
#解法2 蔡勒（Zeller）公式
class Solution:
    def dayOfTheWeek(self, d: int, m: int, y: int) -> str:
        item = {0:"Sunday", 1:"Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday", 5:"Friday", 6:"Saturday"}
        if m == 1:
            m = 13
            y -= 1
        elif m == 2:
            m = 14
            y -= 1
        w = y%100 + (y%100) // 4 + y//100//4 - 2*(y // 100) + 26*(m + 1)//10 + d - 1
        return item[w%7]
```

## 1189.“气球”的最大数量

```python
def maxNumberOfBalloons(self, text: str) -> int:
    c = Counter(text)
    ans = 2**32 -1
    t = "balon"
    print(c)
    for i in t:
        if i == "l" or i == "o":
            if c[i]//2 < ans:
                ans = c[i]//2
        else:
            if c[i] < ans:
                ans = c[i]
    return ans
```

## 1200.最小绝对差

```python
#应用pairwise函数
def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
    m, ans = inf, []
    for a, b in pairwise(sorted(arr)):
        if (cur := b - a) < m:
            m, ans = cur, [[a, b]]
        elif cur == m:
            ans.append([a, b])
    return ans
```

## 1207.独一无二的出现次数

```python
def uniqueOccurrences(self, arr: List[int]) -> bool:
    t = Counter(arr).values()
    return len(t) == len(set(t))
```

## 1217.玩筹码

```python
def minCostToMoveChips(self, position: List[int]) -> int:
    even = 0
    for i in position:
        if not i%2:
            even += 1
    odd = len(position) - even
    return min(odd , even)
```

## 1221.分割平衡字符串

```python
# 字符串切分后去重的两种方法
#list(filter(None,s.split("L")))
#[x for x in s.split(',') if x]

#解法1 栈
def balancedStringSplit(self, s):
    ret, stack = 0, []
    for i in s:
        if not stack or stack[-1] == i:
            stack.append(i)
        else:
            stack.pop()
        if not stack:
            ret += 1
    return ret

#解法2 贪心法                              
def balancedStringSplit(self, s: str) -> int:
    left, right = 0, 0
    ans = 0
    for ch in s:
        if ch == 'L':
            left += 1
        elif ch == 'R':
            right += 1
        if left == right:
            ans += 1
    return ans
```

## 1232.缀点成线

```python
#解法1 combinations函数
def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
    t = list(abs((y2-y1)/(x2-x1)) if x2!=x1 else float("inf") for (x1,y1),(x2,y2) in combinations(coordinates, 2))
    return len(set(t)) == 1

#解法2 zip函数
def checkStraightLine(self, A: List[List[int]]) -> bool:
    seen = set()
    for [x1, y1], [x2, y2] in zip(A, A[1:]):
        if x1 == x2: k = float("inf")
        else: k = (y2-y1) / (x2-x1)
        seen.add(k)
        if len(seen) == 2: return False
    return True
```

## 1252.奇数值单元格的数目

```python
#模拟
def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
    row, col = [0] * m, [0] * n

    for r, c in indices:
        row[r] += 1
        col[c] += 1
    
    a = sum(r % 2 == 1 for r in row)
    b = len(row) - a
    c = sum(c % 2 == 1 for c in col)
    d = len(col) - c
    return a * d + b * c

#容斥原理
def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
    rows, cols = [0] * m, [0] * n
    for r, c in indices:
        rows[r] ^= 1
        cols[c] ^= 1
    return (r := sum(rows)) * n + (c := sum(cols)) * m - 2 * r * c
```

## 1260.二维网格迁移

```python
def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
    nu = []
    ct = 0
    for i in grid:
        for j in i:
            nu.append(j)
            ct += 1
    
    while k:
        t = nu[-1]
        nu.insert(0,t)
        nu.pop()
        k -= 1
    
    ans = []
    m = len(grid)
    n = ct // m
    for i in range(0, len(nu), n):
        ans.append(nu[i:i+n])
    return ans
```

## 1266.访问所有点的最小时间

```python
def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
    ans =  0
    for i in range(len(points)-1):
        ans += max(abs(points[i][1] - points[i+1][1]), abs(points[i][0] - points[i+1][0]))
    return ans
```

## 1275.找出井字棋的获胜者

```python
#数独矩阵
def tictactoe(self, moves: List[List[int]]) -> str:
    # 3阶数独矩阵
    SUDOKU=[[6,1,8],
            [7,5,3],
            [2,9,4]]
    # 如果其中3数和为15就赢
    if 15 in [sum(k) for k in combinations([SUDOKU[i][j] for i,j in moves[::2]], 3)]:return "A"
    if 15 in [sum(k) for k in combinations([SUDOKU[i][j] for i,j in moves[1::2]], 3)]:return "B"
    return "Draw" if len(moves)==9 else "Pending"
```

## 1281.整数的各位积和之差

```python
def subtractProductAndSum(self, n: int) -> int:
    sm = 0
    mult = 1
    while n:
        t = n % 10
        sm += t
        mult *= t
        n //= 10
    return mult - sm
```

## 1287.有序数组中出现次数超过25%的元素

```python
def findSpecialInteger(self, arr: List[int]) -> int:
    dic = Counter(arr)
    t = dic.most_common(1)
    return t[0][0]
```

## 1290.二进制链表转整数

```python
def getDecimalValue(self, head: ListNode) -> int:
    ans = 0
    while head:
        ans = ans *2 + head.val
        head = head.next
    return ans
```

## 1295.统计位数为偶数的数字

```python
def findNumbers(self, nums: List[int]) -> int:
    ans = 0
    for i in nums:
        if not len(str(i)) % 2:
            ans += 1
    return ans
```

## 1299.将每个元素替换为右侧最大元素

```python
def replaceElements(self, arr: List[int]) -> List[int]:
    max = -1
    arr = arr[::-1]
    ans = []
    for i in arr:
        ans.append(max)
        if i > max:
            max = i
    return ans[::-1]
```

## 1304.和为零的N个不同整数

```python
def sumZero(self, n: int) -> List[int]:
    ans = []
    if n%2:
        ans.append(0)
    k = n//2
    for i in range(1,k+1):
        ans.append(i)
        ans.append(-i)
    return ans
```

## 1309.解码字母到整数映射

```python
#正则表达式
def freqAlphabets(self, s: str) -> str:
    return ''.join(chr(int(i[:2]) + 96) for i in re.findall(r'\d\d#|\d', s)) #A[ : 2]:表示索引 0至1行；
```

## 1313.解压缩编码列表

```python
def decompressRLElist(self, nums: List[int]) -> List[int]:
    ans = []
    for i in range(0,len(nums),2):
        frac = nums[i]
        val = nums[i+1]
        while frac:
            ans.append(val)
            frac -= 1
    return ans
```

## 1317.将整数转换为两个无零整数的和

```python
def getNoZeroIntegers(self, n: int) -> List[int]:
    m=1
    while True:
        if '0' not in str(n-m) and '0' not in str(m):
            return [m,n-m]
        else:
            m+=1
```

## 1323.6和9组成的最大数字

```python
'''
str.replace(old, new[, max])
参数
old -- 将被替换的子字符串。
new -- 新字符串，用于替换old子字符串。
max -- 可选字符串, 替换不超过 max 次。
'''
def maximum69Number (self, num: int) -> int:
    i =  str(num).replace("6","9",1)
    return int(i)
```

## 1331.数组序号转换

```python
S'''
enumerate(iterable, start=0)
参数
iterable — —必须是一个序列，或 iterator，或其他支持迭代的对象。
start— —下标起始位置
'''
def arrayRankTransform(self, arr):
    num_map = {num:idx  for idx,num in enumerate(sorted(set(arr)),1)}
    return [num_map[num] for num in arr]
```

## 1332.删除回文子序列

```python
def removePalindromeSub(self, s: str) -> int:
    return (s != s[::-1]) + 1
```

## 1337.矩阵中战斗力最弱的K行

```python
#key + lambda表达式
def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
    return sorted([i for i in range(len(mat))], key=lambda x:sum(mat[x]))[:k]
```

## 1342.将数字变成0的操作次数

```python
def numberOfSteps(self, num: int) -> int:
    cnt = 0
    while num:
        if not num % 2:
            num //= 2
        else:
            num -= 1
        cnt += 1
    return cnt
```

## 1346.检查整数及其两倍数是否存在

```python
def checkIfExist(self, arr: List[int]) -> bool:
    for i in arr:
        if i/2 in arr or i*2 in arr:
            if i != 0:
                return True
            else:
                return Counter(arr)[0] >= 2
    return False
```

## 1351.统计有序矩阵中的负数

```python
def countNegatives(self, grid: List[List[int]]) -> int:
    cnt = 0
    for i in grid:
        for j in i[::-1]:
            if j >= 0:
                break
            else:
                cnt += 1
    return cnt
```

## 1356.根据数字二进制下1的数目排序

```python
def sortByBits(self, arr: List[int]) -> List[int]:
    # 使用哈希表（字典），建立 <二进制1的次数, [数字1， 数字2，...]>的映射
    d = dict()
    for x in arr:
        b = bin(x)  # 把 x 转出二进制字符串
        cnt = b.count('1')  # 统计二进制中 '1' 的个数
        # print('cnt = ', cnt)
        if cnt not in d:
            # 不存在值为 cnt 的key，建立 <cnt, []>的映射关系
            d[cnt] = list()
        d[cnt].append(x)
    # 将字典 d 按 key（1的出现次数）进行排序
    keys = sorted(d.keys())
    ans = list()
    for k in keys:
        # 对 list 进行排序
        ans += sorted(d[k])
    return ans
```

## 1360.日期之间隔几天

```python
#解法1 运用库函数 datetime
from datetime import datetime
class Solution:
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        return  abs((datetime.strptime(date1,'%Y-%m-%d') - datetime.strptime(date2,'%Y-%m-%d')).days)
    
#解法2  zeller公式
def daysBetweenDates(self, date1, date2):
    def toDay(dateStr):
        year = int(dateStr[:4])
        month = int(dateStr[5:7])
        day = int(dateStr[-2:])
        if month <= 2:
            year -= 1
            month += 10
        else:
            month -= 2
        return 365 * year + year // 4 - year // 100 + year // 400 + 30 * month + (3 * month - 1) // 5 + day #- 584418
    return abs(toDay(date1) - toDay(date2))
```

## 1365.有多少小于当前数字的数字

```python
def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
    tp = sorted(nums)[:]
    ans = []
    for i in range(len(nums)):
        ans.append(tp.index(nums[i]))
    return ans
```

## 1370.上升下降字符串

```python
def sortString(self, s: str) -> str:
    str_counter = collections.Counter(s)
    result = []
    flag = False
    while str_counter:
        keys = list(str_counter.keys())
        keys.sort(reverse=flag)
        flag = not flag
        result.append(''.join(keys))
        str_counter -= collections.Counter(keys)
    return ''.join(result)
```

## 1374.生成每种字符都是奇数个的字符串

```python
def generateTheString(self, n: int) -> str:
    if n== 0:
        return ""
    elif n % 2:
        return "a"*n
    else:
        return "a"*(n-1) +"b"
```

## 1380.矩阵中的幸运数

```python
def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
    return list(set(min(row)  for row in matrix) & set(max(col)  for col in zip(*matrix)))
```

## 1385.两个数组间的距离感

```python
'''
bisect是python内置模块，用于有序序列的插入和查找。
查找： bisect(array, item)
插入： insort(array,item)
'''
def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
    arr2.sort()
    res = 0
    for i in arr1:
        left = bisect.bisect_left(arr2, i- d)
        right = bisect.bisect(arr2, i + d)
        if left == right:res += 1
    return res 
```

## 1389.按既定顺序创建目标数组

```python
def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
    ans = []
    for i in range(len(index)):
        ans.insert(index[i], nums[i])
    return ans
```

## 1394.找出数组中的幸运数

```python
def findLucky(self, arr: List[int]) -> int:
    ans = - 1
    for it in Counter(arr).items():
        i,cnt = it[0],it[1]
        if i == cnt and cnt > ans:
            ans = cnt
    return ans
```

## 1399.统计最大组的数目

```python
def countLargestGroup(self, n: int) -> int:
    d={}
    for i in range(1,n+1):
        k=0
        for j in str(i):
            k+=int(j)#转为字符串后得位数和
        if k in d:
            d[k]+=1
        else:
            d[k]=1
    return list(d.values()).count(max(d.values()))		#统计字典中次数最多的个数
```

## 1403.非递增顺序的最小子序列

```python
def minSubsequence(self, nums: List[int]) -> List[int]:
    nums.sort(reverse = True)
    sm = sum(nums)
    ans = []
    ct = 0
    for i in nums:
        ct += i
        ans.append(i)
        if ct > sm//2:
            break
    return ans
```

## 1408.数组中的字符串匹配

```python
def stringMatching(self, words):
    als=','.join(words)
    res=[]
    for s in words:
        if als.count(s)!=1:
            res.append(s)
    return res
```

## 1413.逐步求和得到正数的最小值

```python
def minStartValue(self, nums: List[int]) -> int:
    ans,tmp = 2**32-1,0
    for num in nums:
        tmp+=num
        #保存在累加和过程中出现的最小值，因为只要确保这个最小值加上Value后大于1即可。
        ans = min(tmp,ans)
    return max(-ans+1,1)
```

## 1417.重新格式化字符串

```python
#解法1 分割
def reformat(self, s: str) -> str:
    arr1, arr2 = [], []
    for i in s:
        if i.isalpha():
            arr1.append(i)
        else:
            arr2.append(i)

    if abs(len(arr1)-len(arr2)) > 1:
        return ''

    if len(arr1) < len(arr2):
        arr1, arr2 = arr2, arr1
    res = ''
    for i in range(len(arr2)):
        res += arr1[i]
        res += arr2[i]
    if len(arr1) > len(arr2):
        res += arr1[-1]
    return res

#解法2 正则表达式 + itertool无限迭代器
1、Itertools.count(start=0, step=1)
创建一个迭代对象，生成从start开始的连续整数，步长为step。
如果省略了start则默认从0开始，步长默认为1
如果超过了sys.maxint，则会移除并且从-sys.maxint-1开始计数。
    例：
    from itertools import *
    for i in izip(count(2,6), ['a', 'b', 'c']):
    print i
    输出为：
    (2, 'a')
    (8, 'b')
    (14, 'c')
    
2、Itertools.cycle(iterable)
创建一个迭代对象，对于输入的iterable的元素反复执行循环操作，内部生成iterable中的元素的一个副本，这个副本用来返回循环中的重复项。
    例：
    from itertools import *
    i = 0
    for item in cycle(['a', 'b', 'c']):
        i += 1
        if i == 10:
            break
        print (i, item)
    输出为：
    (1, 'a')
    (2, 'b')
    (3, 'c')
    (4, 'a')    
    (5, 'b')
    (6, 'c')
    (7, 'a')
    (8, 'b')
    (9, 'c')
    
3、Itertools.repeat(object[, times])
创建一个迭代器，重复生成object，如果没有设置times，则会无线生成对象。
    例：
    from itertools import *
    for i in repeat('kivinsae', 5):
        print I
    输出为：
    kivinsae
    kivinsae
    kivinsae
    kivinsae
    kivinsae

    
class Solution:
    def reformat(self, s: str) -> str:
        a=re.findall(r'\d',s)
        b=re.findall(r'[a-z]',s)
        if abs(len(a)-len(b))>1:
            return ''
        a,b=sorted([a,b],key=len)
        return ''.join(map(''.join,itertools.zip_longest(b,a,fillvalue='')))
```

## 1422.分割字符串的最大得分

```python
def maxScore(self, s: str) -> int:
    return max(s[:i].count("0")+s[i:].count("1") for i in range(1,len(s)))
```

## 1431.拥有最多糖果的孩子

```python
def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
    m = max(candies)
    ans = []
    for i in candies:
        if i + extraCandies >= m:
            ans.append(True)
        else:
            ans.append(False)
    return ans
```

## 1436.旅行终点站

```python
def destCity(self, paths: List[List[str]]) -> str:
    zp = list(zip(*paths))
    start = zp[0]
    end = zp[1]
    for i in end:
        if i not in start:
            return i
    return ""
```

## 1437.是否所有1都至少相隔k个元素

```python
#双指针
def kLengthApart(self, nums: List[int], k: int) -> bool:
    l,r= -1,0
    le = len(nums)
    while r < le:
        if nums[r] == 1:
            if l == -1:
                l = r
            else:
                if r - l <= k:
                    return False
                l = r
        r += 1
    return True
```

## 1441.用栈操作构建数组

```python
def buildArray(self, target: List[int], n: int) -> List[str]:
    number = 0#使用一个数，记录每次的栈顶元素
    list1 = []
    for i in target:
        list1.append("Push")
        for j in range(i-number-1):#简单模拟一下，每次进入一个数字，如果不是对应队列中的数字，那么将其插入后弹出即可。
            list1.append("Pop")
            list1.append("Push")
        number = i
    return list1
```

## 1446.连续字符

```python
def maxPower(self, s: str) -> int:
    l = r = ans = 0
    while l < len(s):
        while r < len(s) and s[r] == s[l]:
            r += 1
        ans = max(ans, r - l)
        l = r
    return ans
```

## 1450.在既定时间做作业的学生人数

```
def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    ans = 0
    for i in range(len(startTime)):
        if startTime[i] <= queryTime and endTime[i] >= queryTime:
            ans += 1
    return ans
```

## 1455.检查单词是否为句中其他单词的前缀

```python
def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
    spl = sentence.split()
    le = len(searchWord)
    for i in range(len(spl)):
        if spl[i][:le] == searchWord:
            return i + 1
    return -1
```

## 1460.通过翻转子数组使两个数组相等

```python
def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
    return sorted(arr) == sorted(target)
```

## 1464.数组中两元素的最大乘积

```python
def maxProduct(self, nums: List[int]) -> int:
    tl = sorted([x-1 for x in nums])
    return max(tl[0]*tl[1] ,tl[-1]*tl[-2])
```

## 1470.重新排列数组

```python
def shuffle(self, nums: List[int], n: int) -> List[int]:
    l = len(nums)//2
    a1 = nums[:l]
    a2 = nums[l:]
    c = zip(a1,a2)
    ans = []
    for i in c:
        ans.append(i[0])
        ans.append(i[1])
    return ans

#用chain
'''
from itertools import chain
a = [1, 2, 3, 4]
b = [‘x’, ‘y’, ‘z’]
for x in chain(a, b):
… print(x)
… 1 2 3 4 x y z
'''
def shuffle(self, nums: List[int], n: int) -> List[int]:
    return list(chain(*zip(nums[:n],nums[n:])))
```

## 1475.商品折扣后的最终价格

```python
def finalPrices(self, prices: List[int]) -> List[int]:
    if len(prices) == 1:
        return prices
    a,b = 0,1
    while a < len(prices)-1:
        if prices[b] <= prices[a]:
            prices[a] -= prices[b]
            a += 1
            b = a + 1
        else:
            b += 1
        if b == len(prices):
            a += 1
            b = a+1
    return prices
```

## 1480.一维数组的动态和

```python
def runningSum(self, nums: List[int]) -> List[int]:
    for i in range(1,len(nums)):
        nums[i] += nums[i-1]
    return nums
```

## 1486.数组异或操作

```python
def xorOperation(self, n: int, start: int) -> int:
    ans = 0
    while n:
        ans ^= start
        n-=1
        start+=2
    return ans
```

## 1491.去掉最低工资和最高工资后的工资平均值

```python
def average(self, salary: List[int]) -> float:
    salary.sort()
    del salary[0]
    del salary[-1]
    return sum(salary)/len(salary)
```

## 1496.判断路径是否相交

```python
#路径存储    
def isPathCrossing(self, path: str) -> bool:
    paths = [(0,0)]
    x,y = 0,0
    for i in path:
        if i == 'N':
            x -= 1
        elif i == 'S':
            x += 1
        elif i == 'E':
            y += 1
        else:
            y -= 1
        if (x,y) in paths:
            return True
        paths.append((x,y))
    return False
```

## 1502.判断能否形成等差数列

```
def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
    arr.sort()
    mi = -1
    for i in range(1,len(arr)):
        if mi == -1:
            mi = abs(arr[i] - arr[i-1])
        elif abs(arr[i] - arr[i-1]) != mi:
            return False
    return True
```

## 1507.转变日期格式

```python
def reformatDate(self, date: str) -> str:
    Mth=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data = date.split()
    day = int(data[0][:-2])
    month = Mth.index(data[1])+1
    year = str(data[2])
    if month < 10:
        month = "0" + str(month)
    if day < 10:
        day = "0" + str(day)
    return year+"-"+str(month)+"-"+str(day)
```

## 1512.好数对的数目

```python
def numIdenticalPairs(self, nums: List[int]) -> int:
    return sum(nums[inx+1:].count(i) for inx, i in enumerate(nums))

#方法2 哈希表
def numIdenticalPairs(self, nums: List[int]) -> int:
    ret, dct = 0, defaultdict(int)
    for i in nums:
        ret, dct[i] = ret+dct[i], dct[i]+1
    return ret
```

## 1518.换酒问题

```python
def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
    ans = 0
    pre = 0 
    while numBottles:
        ans += numBottles
        numBottles += pre
        pre =  numBottles% numExchange
        numBottles //= numExchange
    return ans
```

## 1523.在区间范围内统计奇数数目

```python
def countOdds(self, low: int, high: int) -> int:
    if low%2 or high%2:
        return (high-low)//2+1
    else:
        return (high-low)//2
```

## 1528.重新排列字符串

```python
#方法1 zip + lambda 表达式排序
def restoreString(self, s: str, indices: List[int]) -> str:
        t = list(zip(s,indices))
        t.sort(key =  x : x[1])
        ans = ""
        for i in t:
            ans += i[0]
        return ans
#方法2  用dict + zip
def restoreString(self, s: str, indices: List[int]) -> str:
    return ''.join([dict(zip(indices, s))[i] for i in range(len(s))])
```

## 1534.统计好三元组

```python
#用itertools的combinations来完成美剧遍历操作
import itertools
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        return sum(abs(i-j) <= a and abs(j-k) <= b and abs(i-k) <= c for i, j, k in itertools.combinations(arr, 3))
```

## 1539.第k个缺失的正整数

```python
#利用set(range(1,max(arr) + k + 1))-set(arr)求出两个set之间的差
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        return sorted(set(range(1,max(arr) + k + 1)) - set(arr))[k-1]
```

## 1544.整理字符串

```python
#模拟栈
def makeGood(self, s: str) -> str:
        res = [s[0]]
        for char in s[1:]:
            if len(res) >= 1 and ord(res[-1])^ord(char) == 32: #abs(ord(res[-1])-ord(char)) == 32
                res.pop()
            else:
                res.append(char)
        return ''.join(res)
```

## 1550.存在连续三个奇数的数组

```python
def threeConsecutiveOdds(self, arr: List[int]) -> bool:
    le = len(arr)
    for i in range(0,le):
        if i+1 < le and i+2 <le and arr[i]%2 and arr[i+1]%2 and arr[i+2]%2:
            return True
    return False
```

## 1556.千位分割数

```python
#方法1
def thousandSeparator(self, n: int) -> str:
    ans = []
    if n == 0: return "0"
    while n:
        k = n%1000
        if k < 100 and n >= 1000:
            k = "0" + str(k)
        else:
            k = str(k)
        ans.append(k)
        n//=1000
    return ".".join(ans[::-1])

#方法2 format函数
3.1415926	 {:.2f}			3.14	保留小数点后两位
3.1415926	 {:+.2f}		+3.14	带符号保留小数点后两位
-1			{:-.2f}			-1.00	带符号保留小数点后两位
2.71828		{:.0f}			3	不带小数
5			{:0>2d}			05	数字补零 (填充左边, 宽度为2)
5			{:x<4d}			5xxx	数字补x (填充右边, 宽度为4)
10			{:x<4d}			10xx	数字补x (填充右边, 宽度为4)
1000000		{:,}			1,000,000	以逗号分隔的数字格式
0.25		{:.2%}			25.00%	百分比格式
1000000000	{:.2e}			1.00e+09	指数记法
13			{:>10d}	        13	右对齐 (默认, 宽度为10)
13			{:<10d}			13	左对齐 (宽度为10)
13			{:^10d}	    	13	中间对齐 (宽度为10)
def thousandSeparator(self, n: int) -> str:
    return format(n,',').replace(',','.')
```

## 1560.圆形赛道上经过次数最多的扇区

```python
def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
    return list(range(rounds[0], rounds[-1] + 1)) if rounds[0] <= rounds[-1] else list(range(1, rounds[-1] + 1)) + list(range(rounds[0], n + 1))
```

## 1566.重复至少K次且长度为M的模式

```python
def containsPattern(self, arr: List[int], m: int, k: int) -> bool:
    for start in range(len(arr)):
        if start+m*k<=len(arr):
            if arr[start:start+m]*k==arr[start:start+m*k]:
                return True
    return False
```

## 1572.矩阵对角线元素的和

```python
def diagonalSum(self, mat: List[List[int]]) -> int:
    n,res = len(mat),0
    for i in range(n):
        res += mat[i][i]
        mat[i][i] = 0		#通过置零操作避免重复添加
        res += mat[i][n-i-1]
    return res
```

## 1576.替换所有的问号

```python
def modifyString(self, s: str) -> str:
    res = list(s)  # 先将字符串转换为字符列表，方便遍历和替换‘？’的值，字符串无法直接改
    n = len(res)
    for i in range(n):
        # 遇到‘?’就进行替换
        if res[i] == '?':
            # 其实没必要遍历所有的不同字母，只需任意遍历三个互不相同的字母，就能保证一定找到一个与前后字符均不相同的字母，以‘xyz’为例
            for b in "xyz":
                # 防止越界，因为‘?’会在第一个位置或者最后一个位置，此时其没有前后位置
                if not (i > 0 and res[i - 1] == b or i < n - 1 and res[i + 1] == b):
                    res[i] = b  # 前后都不是b,替换为b
                    break  # 替换完，结束内部循环
    return ''.join(res)  # 列表重新转化为字符串
```

## 1582.二进制矩阵中的特殊位置

```python
def numSpecial(self, mat: List[List[int]]) -> int:
    cols = list(zip(*mat))
    res = 0
    for rows in mat:
        if sum(rows) == 1:
            j = rows.index(1)		#index函数查找数组中某个元素的位置
            if sum(cols[j]) == 1:
                res += 1
    return res
```

## 1588.所有奇数长度子数组的和

```python
#解法1
def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    alen = len(arr)
    res = 0
    for i in range(alen):
        if i % 2 == 0:
            for j in range(alen-i):
                res += sum(arr[j:j+i+1])
    return res

#解法2 第二个数的出现的次数 = 第一个数的出现的次数 - 以第一个数结尾的次数 + 以第二个数开头的次
def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    # 长度为n的数组，第一位出现在多少个奇数长度的子数组？
    # 1, 3, 5, ..., length-1/length
    # (length + 1)//2
    n = len(arr)
    l, r, ans, times = 0, n - 1, 0, (n+1) // 2
    while l <= r:
        # 对称性，前后对称位置出现的次数一样
        if l < r:
            ans += times * (arr[l] + arr[r])
        else:
            ans += times * arr[l]
        l += 1
        r -= 1
        # 下一个数比前一个数多了后一个数构成的不带前一个数的奇数子数组的个数
        times += (n - l + 1) // 2 
        # 下一个数比前一个数少了前一个数构成的不带后一个数的奇数子数组的个数
        times -= (l + 1) // 2
    return ans
```

## 1592.重新排列单词间的空格

```python
#count()：返回字符串中某字符数量 divmod():返回商和余数 join()：在一列list的每个元素后面连接前面的变量
def reorderSpaces(self, text: str) -> str:
    c = text.count(" ")
    li = text.strip().split()
    if len(li) == 1:
        return li[0] + " " * c
    s, s1 = divmod(c, len(li) - 1) 
    return (" " * s).join(li)  + " " * s1
```

## 1598.文件夹操作日志搜集器

```python
#模拟栈
def minOperations(self, logs: List[str]) -> int:
    ans = []
    for i in logs:
        if i == "../":
            if len(ans):
                ans.pop()
        elif i != "./":
            ans.append(i)
    return len(ans)
```

## 1603.设计停车系统

```python
class ParkingSystem:
    def __init__(self, big: int, medium: int, small: int):
        self.num_dic = {1:big,2:medium,3:small}

    def addCar(self, carType: int) -> bool:
        if self.num_dic[carType] > 0:
            self.num_dic[carType] -=1
            return True
        return False
```

## 1608.特殊数组的特征值

```python
def specialArray(self, nums: List[int]) -> int:
    if nums == []:
        return 0
    n = len(nums)
    sort_nums = sorted(nums , reverse = True)
    if sort_nums[-1] >= n:
        return n
    
    for i in range(1, n):
        # 如果 nums 是特殊数组，那么其特征值 x 是 唯一的 
        if sort_nums[i] < i <= sort_nums[i - 1]:
            return i
    return -1
```

## 1614.括号的最大嵌套深度

```python
def maxDepth(self, s: str) -> int:
    res = 0
    left = 0
    for c in s:
        if c == '(':
            left += 1
            res = max(res, left)
        elif c == ')':
            left -= 1
    
    return res
```

## 1619.删除某些元素后的数组均值

```python
def trimMean(self, arr: List[int]) -> float:
    arr.sort()
    n = int(len(arr) * 0.05)
    return 1.0*sum(arr[n:n*(-1)])/(len(arr)-2*n)
```

## 1624.两个相同字符之间的最大子字符串

```python
def maxLengthBetweenEqualCharacters(self, s: str) -> int:
    ans = - 1
    for i in range(len(s)-1):
        if s.rfind(s[i],i+1) != -1:
            t = s.rfind(s[i],i+1) - i - 1
            if t > ans:
                ans = t
    return ans
```

## 1629.按键持续时间最长的键

```python
def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
    a1=int(releaseTimes[0])
    a2=0
    for i in range(0,len(releaseTimes)-1):
        if int(releaseTimes[i+1])-int(releaseTimes[i])>a1:
            a1=int(releaseTimes[i+1])-int(releaseTimes[i])
            a2=i+1
        elif int(releaseTimes[i+1])-int(releaseTimes[i])==a1:
            if keysPressed[i+1]>keysPressed[a2]:
                a2=i+1
    return keysPressed[a2]
```

## 1636.按照频率将数组升序排序

```python
'''
list.sort(key=lambda x: (-len(x), x), reverse=True)
将list首先根据长度排列，再按照abc字典顺序排列，最后按照降序排列。

解释：reverse=True按照降序排列，reverse()为将整个列表反过来；
lambda x是指具有排序规则：按照字母从小到大排序，或者按照字符串长度排序；
(-len(x), x)是指首先用x的长度排序，如果长度相同则用出现的先后排序；
两个排序原则，按照-len(x)排序，len(x)表明将长度从小到大排序，那么-len(x)，表明将字符串从大到小排序，如果出现两个字符串长度相同的情况，按照第二个x，也就是按照x的大小，从小到大排序。
'''
def frequencySort(self, nums: List[int]) -> List[int]:
    # c = Counter(nums)
    # cc = sorted(c.items(),key=lambda x:x[1]) #按照值从小到大排序 Counter排序方法
    return sorted(nums, key=lambda x: (nums.count(x), -x))
```

## 1640.能否连接形成数组

```python
def canFormArray(self, arr, pieces):
    p_hash = {}
    for i, v in enumerate(pieces):
        p_hash.setdefault(v[0], i)
    res = []
    for a in arr:
        if a in p_hash:
            res.extend(pieces[p_hash.get(a)])
    return arr == res
```

## 1646.获取生成数组中的最大值

```python
def getMaximumGenerated(self, n: int) -> int:
    ans = 1
    nums = [0,1]
    if n == 0:
        return 0
    i = 2
    while i <= n:
        k = -1
        if i%2:
            k = nums[i//2] + nums[i//2 + 1]
        else:
            k = nums[i//2]
        nums.append(k)
        i += 1
        if k > ans : ans = k
    return ans
```

## 1652.拆炸弹

```python
def decrypt(self, code: List[int], k: int) -> List[int]:
    ret = []
    extend_code =  code + code
    n = len(code)  
    for i in range(n):
        start = i+1  if k >= 0 else n+k+i
        end = start+k if k >= 0 else n+i 
        num =  0 if k == 0 else sum(extend_code[start:end])
        ret.append(num)
    return ret
```

## 1656.设计有序流

```python
class OrderedStream:
    def __init__(self, n: int):
        self.data = ["" for _ in range(n + 1)]
        self.n = n
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> List[str]:
        self.data[idKey] = value
        res = []

        i = self.ptr
        while i <= self.n and self.data[i] != "":
            res.append(self.data[i])
            i += 1

        self.ptr = i
        return res
```

## 1662.检查两个字符数组是否相等

```python
def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
    s1 = "".join(word1)
    s2 = "".join(word2)
    return s1 == s2
```

## 1668.最大重复子字符串

```python
def maxRepeating(self, sequence: str, word: str) -> int:
    dp=[0]*(len(sequence)+1)
    k=len(word)
    for i in range(k,len(sequence)+1):
        if sequence[i-k:i]==word:
            dp[i]=dp[i-k]+1
    return max(dp)
```

## 1672.最富有客户的资产总量

```python
def maximumWealth(self, accounts: List[List[int]]) -> int:
    accounts.sort(key = lambda x: sum(x), reverse=True)
    return sum(accounts[0])
```

## 1678.设计Goal解析器

```python
def interpret(self, command: str) -> str:
    command = command.replace("(al)", "al")
    command = command.replace("()", "o")
    return command
```

## 1681.统计一致字符串的数目

```python
def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
    ans = 0
    for i in words:
        for j in i:
            if not j in allowed:
                ans-=1
                break
        ans += 1
    return ans
```

## 1688.比赛中的配对次数

```python
def numberOfMatches(self, n: int) -> int:
    ans = 0
    while n > 1:
        ans += n//2
        if n%2:
            n = n//2 +1
        else:
            n //= 2
    return ans
```

## 1694.重新格式化

```python
def reformatNumber(self, number: str) -> str:
    if not number:
        return number
    number = number.replace("-", "").replace(" ", "")
    # number = [i for i in str(number) if i.isdigit()]
    r = ["".join(number[i: i+3]) for i in range(0, len(number), 3)]
    if len(r[-1]) == 1:
        r[-2], r[-1] = r[-2][:-1], r[-2][-1] + r[-1]
    return "-".join(r)
```

## 1700.无法吃午餐的学生数量

```python
def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    for sandwich in sandwiches:
        if sandwich in students:
            students.pop(students.index(sandwich))
        else:
            return len(students)
    return 0 
```

## 1704.判断字符串的两半是否相似

```python
def halvesAreAlike(self, s: str) -> bool:
    vow = ['a','e','i','o','u','A','E','I','O','U']
    le = len(s)
    s1 = s[:le//2]
    s2 = s[le//2:]
    k1 = 0
    k2 = 0
    for i in s1:
        if i in vow:
            k1+=1
    for j in s2:
        if j in vow:
            k2+=1
    return k1==k2

#精简写法
def halvesAreAlike(self, s: str) -> bool:
    mid, ret = len(s)//2, 0
    for i, n in enumerate(s):
        if n in ['a','e','i','o','u','A','E','I','O','U']:
            ret += 1 if i < mid else -1
    return not ret
```

## 1710.卡车上的最大单元数

```python
def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    num = 0
    for box in sorted(boxTypes,key = lambda x:x[1],reverse = True):
        num += min(box[0],truckSize) * box[1]
        truckSize -= min(box[0],truckSize)
        if truckSize == 0:
            return num
    return num
```

## 1716.计算力扣银行的钱

```python
#方法1 模拟
def totalMoney(self, n: int) -> int:
    week = 1
    ans = 0
    while n > 7:
        ans += (21 + week*7)
        n -= 7
        week += 1
    ans += (2*week+n-1)*n//2
    return ans
#方法2 divmod函数
def totalMoney(self, n):
    total = 0
    week, day = divmod(n, 7)
    # total += 28 * week + (week - 1) * week * 7 // 2 
    total += week * (week + 7) * 7 // 2 # 合并后
    total += (week + 1  + week + day ) * day // 2
    return total
```

## 1720.解码异或后的数组

```python
def decode(self, encoded: List[int], first: int) -> List[int]:
    n = len(encoded) + 1
    decode = [0] * n
    decode[0] = first
    # 请注意数组下标的调整
    for i in range(1, n):
        decode[i] = encoded[i-1] ^ decode[i-1]
    return decode
```

## 1725.可以形成最大正方形的矩形数目

```python
def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
    ans = []
    for i in rectangles:
        ans.append(min(i))
    return ans.count(max(ans))
```

## 1732.找到最高海拔

```python
def largestAltitude(self, gain: List[int]) -> int:
    mx = 0
    ct = 0
    for i in gain:
        ct += i
        if ct > mx:
            mx = ct
    return mx
```

## 1736.替换隐藏数字得到的最晚时间

```python
#枚举法
def maximumTime(self, time: str) -> str:
    time = list(time)
    for i in range(len(time)): 
        if time[i] == "?": 
            if i == 0: 
                time[i] = "2" if time[i+1] in "?0123" else "1"
            elif i == 1: 
                time[i] = "3" if time[0] == "2" else "9"
            elif i == 3: 
                time[i] = "5"
            else: 
                time[i] = "9"           
    return "".join(time)
```

## 1742.盒子中小球的最大数量

```python
def countBalls(self, lowLimit: int, highLimit: int) -> int:
    dic = {}
    for i in range(lowLimit,highLimit+1):
        tp = 0
        while i >= 10:
            tp += i%10
            i//=10
        tp += i
        if tp in dic.keys():
            dic[tp] += 1
        else:
            dic[tp] = 1
    return max(dic.values())   
```

## 1748.唯一元素的和

```python
#Counter函数
def sumOfUnique(self, nums: List[int]) -> int:
    return sum(num for num, cnt in Counter(nums).items() if cnt == 1)
```

## 1752.检查数组是否经排序和轮转得到

```python
#有序数组轮转数字相邻的两次，相同一位前面的只会比后面的出现一次大于的情况
def check(self, nums: List[int]) -> bool:
    return sum(a > b for a, b in zip(nums, nums[1:] + nums[:1])) <= 1
```

## 1758.生成交替二进制字符串的最少操作数

```python
'''
最终生成的数字要么是 '01010101...' 或者 '10101010...'
用数学分析为
奇数为 1 偶数为 0 n1
奇数为 0 偶数为 1 n2
其中通过分析可知 n1 + n2 =len(s)
'''
def minOperations(self, s: str) -> int:
    n = len(s)
    res = sum(i % 2 == int(s[i]) for i in range(n))
    return res if res <= n - res else n - res
```

## 1763.最长的美好子字符串

```python
#分治思想
def longestNiceSubstring(self, s: str) -> str:
    if len(s) < 2:
        return ""
    for i, c in enumerate(s):
        # 存在任意不满足题目的割点
        if c.upper() not in s or c.lower() not in s:
            return max(self.longestNiceSubstring(s[:i]), self.longestNiceSubstring(s[i+1:]), key = len)
    return s
```

## 1768.交替合并字符串

```python
#使用zip_longest函数
#zip_longest 与 zip 函数唯一不同的是如果两个可迭代参数长度不同时，按最长的输出，长度不足的用 fillvalue 进行代替，默认为 None
def mergeAlternately(self, word1: str, word2: str) -> str:
    return "".join(sum(zip_longest(word1, word2, fillvalue=""), ()))
'''
尽管sum()主要用于对数值进行操作，但您也可以使用该函数来连接列表和元组等序列。为此，您需要为 提供适当的值start：
>>>
>>> num_lists = [[1, 2, 3], [4, 5, 6]]
>>> sum(num_lists, start=[])
[1, 2, 3, 4, 5, 6]
 
>>> # Equivalent concatenation
>>> [1, 2, 3] + [4, 5, 6]
[1, 2, 3, 4, 5, 6]
 
>>> num_tuples = ((1, 2, 3), (4, 5, 6))
>>> sum(num_tuples, start=())
(1, 2, 3, 4, 5, 6)
 
>>> # Equivalent concatenation
>>> (1, 2, 3) + (4, 5, 6)
(1, 2, 3, 4, 5, 6)
'''
#deque函数 模拟栈
def mergeAlternately(self, word1: str, word2: str) -> str:
    a, b = deque(word1), deque(word2)
    res = ""
    while a and b:
        res += a.popleft()
        res += b.popleft()
    res += "".join(a)+"".join(b)
    return res
```

## 1773.统计匹配检索规则的物品数量

```python
def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
    dic ={"type":0, "color":1, "name":2}
    ans = 0
    for i in items:
        if i[dic[ruleKey]] == ruleValue:
            ans += 1
    return ans
```

## 1779.找到最近的有相同X或Y坐标的点

```python
def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
    a=-1
    b=float('inf')
    for i in range(0,len(points)):
        if(points[i][0]==x or points[i][1]==y):
            c=abs(points[i][0]-x)+abs(points[i][1]-y)
            if(c<b):
                a=i 
                b=c
    return a
```

## 1784.检查二进制字符串字段

```python
def checkOnesSegment(self, s: str) -> bool:
    return '01' not in s
```

## 1790.仅执行一次字符串交换能否使两个字符串相等

```python
def areAlmostEqual(self, s1: str, s2: str) -> bool:
    return s1 == s2 or Counter(s1) == Counter(s2) and sum(s1[i] != s2[i]  for i in range(len(s1))) == 2
```

## 1791.找出星型图的中心节点

```python
#集合的交集
def findCenter(self, edges: List[List[int]]) -> int:
    return set(edges[0]).intersection(set(edges[1])).pop()
```

## 1796.字符串中第二大的数字

```python
def secondHighest(self, s: str) -> int:
    res = set(int(char)  for char in s if char.isdigit())
    return -1 if len(res) <= 1 else sorted(res)[-2]
```

## 1800.最大升序子数组和

```python
def maxAscendingSum(self, nums: List[int]) -> int:
    index = 1
    mx = 0
    le = len(nums)
    cnt = nums[0]
    while index < le and index > 0:
        if nums[index] <= nums[index-1]:
            if cnt > mx : mx= cnt
            cnt = nums[index]
        else:
            cnt+=nums[index]
        index+=1
    return max(mx,cnt)
```

## 1805.字符串中不同整数的数目

```python
def numDifferentIntegers(self, word: str) -> int:
    wl = list(word)
    le = len(wl)
    for i in range(le):
        if wl[i].isalpha():
            wl[i] = " "
    t = "".join(wl).split()
    ans = []
    for i in t:
        ans.append(int(i))
    return len(set(ans))

#解法2 正则表达式
def numDifferentIntegers(self, word: str) -> int:
    res = re.split(r'[a-z]+', word)
    # return len(set([int(i) for i in re.findall(r"[0-9]+", word)]))
    return len(set([int(i) for i in res if i != '']))
```

## 1812.判断国际象棋棋盘中一个格子的颜色

```python
def squareIsWhite(self, coordinates: str) -> bool:
    return abs(ord(coordinates[0])- int(coordinates[1])) % 2 != 0
```

## 1816.截断句子

```python
def truncateSentence(self, s: str, k: int) -> str:
    return " ".join(s.split()[:k])
```

## 1822.数组元素积的符号

```python
def arraySign(self, nums: List[int]) -> int:
    ans = 1
    for i in nums:
        ans *=i
    return ans//abs(ans) if ans!=0 else 0

#解法2 reduce函数
def arraySign(self, nums: List[int]) -> int:
    if 0 in nums:
        return 0
    return 1 if reduce(lambda x,y: x*y, nums) > 0 else -1
```

## 1827.最少操作使数组递增

```python
def minOperations(self, nums: List[int]) -> int:
    le = len(nums)
    ans = 0
    if le <= 1: return 0
    for i in range(1,le):
        if nums[i] <= nums[i-1]:
            ans += nums[i-1] -nums[i] + 1
            nums[i] = nums[i-1] + 1
    return ans
```

## 1832.判断句子是否为全字母句

```python
def checkIfPangram(self, sentence: str) -> bool:
    return len(Counter(sentence).keys()) == 26
```

## 1837.K进制表示下的各位数字总和

```python
def sumBase(self, n: int, k: int) -> int:
    ans = 0
    while n:
        ans += n%k
        n //= k
    return ans
```

## 1844.将所有数字用字符替换

```python
def replaceDigits(self, s: str) -> str:
    ls = list(s)
    le = len(ls)
    for i in range(1,le,2):
        ls[i] = chr(ord(ls[i-1]) + int(ls[i]))
    return "".join(ls)
```

## 1848.到目标元素的最小距离

```python
 def getMinDistance(self, nums: List[int], target: int, start: int) -> int:
    cnt = 0
    l,r= start,start
    le = len(nums)
    while l >= 0 or r < le:
        if (l >=0 and nums[l] == target) or (r < le and nums[r] == target):
            break
        cnt += 1
        l -= 1
        r += 1
    return cnt
```

## 1854.人口最多的年份

```python
def maximumPopulation(self, logs: List[List[int]]) -> int:
    ans  = [0]*105
    for i in logs:
        s,e = i[0],i[1]
        for j in range(s-1950,e-1950):
                ans[j] += 1
    return 1950 + ans.index(max(ans))
```

## 1859.将句子排序

```python
def sortSentence(self, s: str) -> str:
    sp = s.split()
    sp.sort(key = lambda x: x[-1])
    return " ".join([i[:-1] for i in sp])
```

## 1863.找出所有子集的异或总和再求和

```python
def subsetXORSum(self, nums: List[int]) -> int:
    return sum(reduce(xor, tup) for i in range(1,len(nums)+1) for tup in combinations(nums,i))
```

## 1869.哪种连续子字符串更长

```python
def checkZeroOnes(self, s: str) -> bool:
    l1 = len(max(s.split("0")))
    l2 = len(max(s.split("1")))
    return l1 > l2
```

## 1876.长度为三且各字符不同的子字符串

```python
def countGoodSubstrings(self, s: str) -> int:
    ans = 0
    st = list(s)
    le = len(st)
    for i in range(0, le-2):
        if len(set(st[i:i+3])) == 3:
            ans += 1
    return ans
```

## 1880.检查某单词是否等于两单词之和

```python
def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
    tp1 =[str(ord(i)-97) for i in firstWord]
    tp2 =[str(ord(i)-97) for i in secondWord]
    tp3 =[str(ord(i)-97) for i in targetWord]
    s1= int("".join(tp1))
    s2= int("".join(tp2))
    s3= int("".join(tp3))
    return s1+s2==s3
```

## 1886.判断矩阵经轮转后是否一致

```python
def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
    if mat == target:
        return True
    target = [tuple(i) for i in target]  # zip转换之后每一个元素都是元组，所以把target每一个元素也转为元组，方便进行比较
    for i in range(3):  # 转3次就够了，转4次就转回来了 
        mat = list(zip(*mat))[::-1] # 先横向变纵向，然后再让顺序颠倒，从而实现矩阵90°旋转
        if mat == target:
            return True 
    return False
```

## 1893.检查是否区域内所有整数都被覆盖

```python
def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
    diff = defaultdict(int)
    for l, r in ranges:
        diff[l] += 1
        diff[r+1] -= 1
    curr = 0
    for i in range(1, right + 1):
        curr += diff[i]
        if curr <= 0 and left <= i:
            return False
    return True
```

## 1897.重新分配字符使所有字符串都相等

```python
def makeEqual(self, words: List[str]) -> bool:
    le = len(words)
    tp = "".join(words)
    z = Counter(tp)
    for i in z.values():
        if i%le:
            return False
    return True
```

## 1903.字符串中的最大奇数

```python
def largestOddNumber(self, num: str) -> str:
    s = list(num[::-1])
    for i in range(len(s)):
        if int(s[i])%2:
            return "".join(s[i:][::-1])
    return ""
```

## 1909.删除一个元素使数组严格递增

```python
#模拟栈
def canBeIncreasing(self, nums: List[int]) -> bool:
    stack=[]
    count=0
    for i in nums:
        if not stack or i>stack[-1]:
            stack.append(i)
        elif i<=stack[-1]:
            tmp=stack.pop()
            count+=1
            if count>1:
                return False
            if not stack or i>stack[-1]:
                stack.append(i)
            elif i<=stack[-1]:
                stack.append(tmp)
    return True
```

## 1913.两个数对之间的最大乘积差

```python
def maxProductDifference(self, nums: List[int]) -> int:
    nums.sort(reverse = True)
    return nums[0]*nums[1] - nums[-2]*nums[-1]
```

## 1920.基于排列构建数组

```python
 def buildArray(self, nums: List[int]) -> List[int]:
    n = nums[:]
    for i in range(len(nums)):
        nums[i] = n[n[i]]
    return nums
```

## 1925.统计平方和三元组的数目

```python
#哈希 + 枚举
def countTriples(self, n: int) -> int:
    dic = dict()
    res = 0
    for i in range(1,n+1):
        dic[i*i] = i
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            if (i*i + j*j) in dic:
                res += 2
    return res
```

## 1929.数组串联

```python
def getConcatenation(self, nums: List[int]) -> List[int]:
    return nums + nums
```

## 1935.可以输入的最大单词数

```python
def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    sp = text.split()
    ans = 0
    for i in sp:
        ans += 1
        for j in brokenLetters:
            if j in i:
                ans -= 1
                break
    return ans
```

## 1941.检查是否所有字符出现次数相同

```python
def areOccurrencesEqual(self, s: str) -> bool:
    return len(set(Counter(s).values()))==1
```

## 1945.字符串转化后的各位数字之和

```python
def getLucky(self, s: str, k: int) -> int:
    def cnt(i):
        ans = 0
        while i:
            ans += i%10
            i//=10
        return ans
    tmp = 0 
    for i in s:
        tmp += cnt(ord(i)-96)
    k -= 1
    while k:
        tmp = cnt(tmp)
        k -= 1
    return tmp
```

## 1952.三除数

```python
def isThree(self, n: int) -> bool:
    cnt = 1
    for i in range(2,n+1):
        if not n%i:
            cnt += 1
            if cnt > 3:
                return False
    return cnt == 3
```

## 1957.删除字符使字符串变好

```python
def makeFancyString(self, s: str) -> str:
    le = len(s)
    if le <= 2:
        return s
    ans = [s[0], s[1]]
    for i in range(2,le):
        if not (s[i] == ans[-1] and s[i] == ans[-2]):
            ans.append(s[i])
    return "".join(ans)
```

## 1961.检查字符串是否为数组前缀

```python
def isPrefixString(self, s: str, words: List[str]) -> bool:
    ts = ""
    for i in words:
        ts += i
        if ts == s:
            return True
    return False
```

## 1967.作为子字符串出现在单词中的字符串数目

```python
def numOfStrings(self, patterns: List[str], word: str) -> int:
    cnt = 0
    for i in patterns:
        if i in word:
            cnt += 1
    return cnt
```

## 1971.寻找图中是否存在路径

```python
#DFS
from collections import defaultdict
def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
    if start==end:
        return True
    #创建图
    paths = defaultdict(set)
    for edge in edges:
        paths[edge[0]].add(edge[1])
        paths[edge[1]].add(edge[0])
    
    visited = set()
    visited.add(start) 
    return self.dfs(paths, start, end, visited)

def dfs(self, paths, start, end, visited):
    if end in paths[start]:
        return True
    res = False
    for newStart in list(paths[start]):
        if newStart in visited: #防止进入死循环
            continue
        visited.add(newStart) 
        res = self.dfs(paths, newStart, end, visited)
        if res: #剪枝：只要遇到end就返回，不再探索其他路径
            return res
    return res
```

## 1974.使用特殊打字机键入单词的最少时间

```python
def minTimeToType(self, word: str) -> int:
    word = "a" + word
    le = len(word)
    ans = le -1
    for i in range(1,le):
        n1,n2 = min(ord(word[i-1])-96,ord(word[i])-96), max(ord(word[i-1])-96,ord(word[i])-96)
        ans += min(n2-n1,abs(n2-n1-26))
    return ans
```

## 1979.找出数组的最大公约数

```python
def findGCD(self, nums: List[int]) -> int:
    mi = min(nums)
    mx = max(nums)
    tp = mi
    while tp:
        if not mx%tp and not mi%tp:
            return tp
        tp-=1
    return 1
```

## 1984.学生分数的最小差值

```python
def minimumDifference(self, nums: List[int], k: int) -> int:
    nums.sort()
    return min(j-i for i, j in zip(nums, nums[k-1:]))
```

## 1991.找到数组的中间位置

```python
def findMiddleIndex(self, nums: List[int]) -> int:
    tar = sum(nums)
    le = len(nums)
    cnt = 0
    for i in range(le):
        if cnt == (tar - nums[i])/2:
            return i
        cnt += nums[i]
    return -1
```

## 1995.统计特殊四元组

```python
#解法1 背包规划
def countQuadruplets(self, nums: List[int]) -> int:
    l, ans = Counter(), 0
    for i in range(1, len(nums) - 2):
        # 到目前为止统计了所有0到i的两坐标和
        for j in range(i):
            l[nums[i] + nums[j]] += 1
        # 目前第三个坐标为i+1，枚举第四个坐标j的范围
        for j in range(i + 2, len(nums)):
            # 叠加以前统计的左半段和的结果，i+1作为第三个idx和j最多组成这么多
            ans += l[nums[j] - nums[i+1]]
    return ans

#解法2
nums[a] + nums[b] + nums[c] == nums[d] 转换为 nums[a] + nums[b] = nums[d] - nums[c] 且 a < b < c < d
通过 freq 字典保存 nums[a] + nums[b] 出现的次数求和 nums[d] - nums[c] 在之前的字典里面出现的次数和
def countQuadruplets(self, nums: List[int]) -> int:
    res = 0 
    n = len(nums)
    freq = defaultdict(int)
    for i in range(n): 
        for j in range(i+1, n): # j > i
            res += freq[nums[j] - nums[i]]
        for k in range(i): # k < i
            freq[nums[k] + nums[i]] += 1
    return res 
```

## 2000.反转单词前缀

```python
def reversePrefix(self, word: str, ch: str) -> str:
    i = word.find(ch)
    return word[:i+1][::-1] + word[i+1:]
```

## 2006.差的绝对值为K的数对数目

```python
def countKDifference(self, nums: List[int], k: int) -> int:
    l = Counter()
    ans = 0
    for i in nums:
        ans += l[i-k]
        ans += l[i+k]
        l[i] += 1
    return ans
```

## 2011.执行操作后的变量值

```python
def finalValueAfterOperations(self, operations: List[str]) -> int:
    x = 0
    for i in operations:
        if "+" in  i:
            x += 1
        else:
            x -= 1
    return x
```

## 2016.增量元素之间的最大差值

```python
#可以在遍历数组时维护已经遍历过的元素中的最小值，并计算当前位置的最大差值
def maximumDifference(self, nums: List[int]) -> int:
    m, ans = inf, 0
    for num in nums:
        m, ans = min(num, m), max(ans, num - m)
    return ans if ans > 0 else -1
```

## 2022.将一维数组转变为二维数组

```python
def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
    ans = []
    le = len(original)
    if le != m*n: return []
    for i in range(0,le,n):
        ans.append(original[i:i+n])
    return ans
```

## 2027.转换字符串的最少操作次数

```python
def minimumMoves(self, s: str) -> int:
    ans = 0
    n = len(s)
    l = 0
    while l < n:
        if s[l] == 'O':
            l += 1
        else:
            l += 3
            ans += 1
    return ans
```

## 2032.至少在两个数组中出现的值

```python
def twoOutOfThree(self, nums1: List[int], nums2: List[int], nums3: List[int]) -> List[int]:
    s1 = set(nums1)
    s2 = set(nums2)
    s3 = set(nums3)
    s12 = s1.intersection(s2)
    s13 = s1.intersection(s3)
    s23 = s2.intersection(s3)
    return list(s12.union(s13.union(s23)))
```

## 2037.使每位学生都有座位的最少移动次数

```python
def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats.sort()
    students.sort()
    le = len(seats)
    ans = 0
    for i in range(le):
        ans += abs(seats[i] - students[i])
    return ans
```

## 2042.检查句子中的数字是否递增

```python
def areNumbersAscending(self, s: str) -> bool:
    #a =re.findall(r'\d\d|\d', s)
    sp =s.split()
    pre = -1
    for i in sp:
        if i.isdigit():
            if int(i) > pre:
                pre = int(i)
            else:
                return False
    return True
```

## 2047.句子中的有效单词数

```python
def countValidWords(self, sentence: str) -> int:
    def helper(word):
        n, appear = len(word), False
        for i, c in enumerate(word):
            if 'a' <= c <= 'z':
                continue
            elif c == '-':
                if appear or not i or i == n - 1 or not ('a' <= word[i-1] <= 'z' and 'a' <= word[i+1] <= 'z'):
                    return False
                appear = True
            elif c in '!.,':
                if i != n - 1:
                    return False
            else:
                return False
        return True 
    return sum(helper(w) for w in sentence.split(' ') if w)
```

## 2053.数组中第K个独一无二的字符串

```python
#lambda表达式
filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回迭代器对象（Python2是列表），可以使用list()转换为列表
list2 = filter(lambda x:x%2==0, [1,2,3,4,5,6])
print(list(list2)) #输出：[2, 4, 6]

map()接收一个函数 f 和一个或多个序列 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 迭代器（Python2是列表） 并返回
list2_1 = map(lambda x,y : x*y, [1,2,3,4,5],[6,7,8,9,10])
print(list(list2_1)) #输出：[6, 14, 24, 36, 50]

reduce()函数对一个数据集合的所有数据进行操作：用传给 reduce 中的函数 function（必须有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果
function -- 函数，有两个参数
iterable -- 可迭代对象
initializer -- 可选，初始参数

使用例子：
from functools import reduce
def add(x, y):
    return x + y
def mulit(x, y):
    return x * y
print(reduce(add, [1, 2, 3, 4, 5])) #输出：15
print(reduce(add, [1, 2, 3, 4, 5], 10)) #输出：25
print(reduce(mulit, [1, 2, 3, 4, 5])) #输出：120
print(reduce(mulit, [1, 2, 3, 4, 5], 10)) #输出：1200
print(reduce(lambda x,y:x+y,[1, 2, 3, 4, 5]))#输出：15
print(reduce(lambda x,y:x+y,[1, 2, 3, 4, 5], 10))#输出：25

#用Counter和lambda表达式
def kthDistinct(self, arr: List[str], k: int) -> str:
    lc = Counter(arr)
    newl = list(filter(lambda x: x[1] == 1, list(lc.items())))
    return "" if len(newl) < k else newl[k-1][0]
```

## 2057.值相等的最小索引

```python
def smallestEqual(self, nums: List[int]) -> int:
    le = len(nums)
    for i in range(le):
        if i % 10 == nums[i]:
            return i
    return -1
```

## 2062.统计字符串中的元音子字符串

```python
#滑动窗口
def countVowelSubstrings(self, word: str) -> int: 
    ans,n = 0,len(word)
    for i in range(n-4):
        for j in range(i,n+1):
            if set(word[i:j]) == set("aeiou") :
                ans += 1
```

## 2068.检查两个字符串是否几乎相等

```python
def checkAlmostEquivalent(self, word1: str, word2: str) -> bool:
    w1 = Counter(word1)
    w2 = Counter(word2)
    for i in w1.items():
        if abs(w2[i[0]] - i[1]) >3:
            return False
    for i in w2.items():
        if abs(w1[i[0]] - i[1]) >3:
            return False
    return True
```

## 2073.买票需要的时间

```python
def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
    time_sum = 0
    for i in range(len(tickets)):
        if i <= k:
            if tickets[i] >= tickets[k]:
                time_sum += tickets[k]
            else:
                time_sum += tickets[i]
        else:
            if tickets[i] >= tickets[k]-1:
                time_sum += tickets[k]-1
            else:
                time_sum += tickets[i]
    return time_sum
```

## 2078.两栋颜色不同且距离最远的房子

```python
def maxDistance(self, colors: List[int]) -> int:
    n = len(colors)
    left, right = 0, n - 1
    while colors[-1] == colors[left]: 
        left += 1
    while colors[0] == colors[right]: 
        right -= 1
    return max(n - left - 1, right)
```

## 2085.统计出现过一次的公共字符串

```python
def countWords(self, words1: List[str], words2: List[str]) -> int:
    c1 = Counter(words1)
    c2 = Counter(words2)
    ans = 0
    for i in c1.items():
        if i[1] == 1 and c2[i[0]] == 1:
            ans += 1
    return ans
```

## 2089.找出数组排序后的目标下标

```python
def targetIndices(self, nums: List[int], target: int) -> List[int]:
    ans = []
    le = len(nums)
    nums.sort()
    for i in range(le):
        if nums[i] == target:
            ans.append(i)
    return ans
```

## 2094.找出3位偶数

```python
#排列组合
'''
itertools.permutations() ，它接受一个集合并产生一个元组序列，每个元组由集合中所有元素的一个可能排列组成。也就是说通过打乱集合中元素排列顺序生成一个元组
itertools.combinations() 可得到输入集合中元素的所有的组合（按序）
itertools.combinations_with_replacement() 允许同一个元素被选择多次
'''
def findEvenNumbers(self, digits: List[int]) -> List[int]:
    # itertools.permutations(digits, 3)求出digits中任意三个数的全排列。
    return sorted({i*100+j*10+k for i, j, k in itertools.permutations(digits, 3) if i != 0 and k%2 == 0})
```

## 2099.找出和最大的长度为K的子序列

```python
def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
    tp = sorted(nums)[-k:]
    c = Counter(tp)
    ans = []
    for i in nums:
        if c[i]:
            ans.append(i)
            c[i]-=1
    return ans
```

## 2103.环和杆

```python
def countPoints(self, rings: str) -> int:
    ans = 0
    for i in range(10):
        if "R"+ str(i) in rings and "B"+ str(i) in rings and "G"+ str(i) in rings:
            ans += 1
    return ans
```

## 2108. 找出数组中的第一个回文字符串

```python
def firstPalindrome(self, words: List[str]) -> str:
    for i in words:
        if i == i[::-1]:
            return i
    return ""
```

## 2114.句子中的最多单词数

```python
def mostWordsFound(self, sentences: List[str]) -> int:
    ans = -1
    for i in sentences:
        le = len(i.split())
        if  le > ans:
            ans = le
    return ans
```

## 2119.反转两次的数字

```python
 def isSameAfterReversals(self, num: int) -> bool:
    return num%10 != 0 or num == 0
```

## 2124.检查是否所有的A都在B之前

```python
def checkString(self, s: str) -> bool:
    return "ba" not in s
```

## 2129.将标题首字母大写

```python
def capitalizeTitle(self, title: str) -> str:
    ans = []
    tp = title.split()
    for i in tp:
        i = i.lower()
        if len(i) > 2:
            i = i.capitalize()
        ans.append(i)
    return " ".join(ans)
```

## 2133.检查是否每一行每一列都包含全部整数

```python
def checkValid(self, matrix: List[List[int]]) -> bool:
    k = list(zip(*matrix))
    n = len(matrix[0])
    for i in matrix:
        if len(set(i)) != n:
            return False
    for i in k:
        if len(set(i)) != n:
            return False
    return True
```

## 2138.将字符串拆分为若干长度为k的组

```python
def divideString(self, s: str, k: int, fill: str) -> List[str]:
    t = 0
    le = len(s)
    if le%k: t = k - le%k
    ans = []
    s += t*fill
    for i in range(0,le,k):
        ans.append(s[i:i+k])
    return ans
```

## 2144.打折购买糖果的最小开销

```python
def minimumCost(self, cost: List[int]) -> int:
    le = len(cost)
    cost.sort(reverse = True)
    cnt = 0
    for i in range(le):
        if not (i+1)%3:
            cnt += cost[i]
    return sum(cost) - cnt
```

## 2148.元素比较

```python
def countElements(self, nums: List[int]) -> int:
    mi = min(nums)
    mx = max(nums)
    cnt = 0
    tn = list(filter(lambda x: x!=mi and x!=mx, nums))
    return len(tn)
```

## 2154.将找到的值乘以2

```python
def findFinalValue(self, nums: List[int], original: int) -> int:
    while original in nums:
        original *= 2
    return original
```

## 2160.拆分数位后四位数字的最小和

```python
def minimumSum(self, num: int) -> int:
    n = sorted([int(i) for i in str(num)])
    return 10*(n[0]+n[1]) + n[2] + n[3]
```

## 2164.对奇偶下标分别排序

```python
#s[1:4:2] 括号里参数意思为：[起始索引：结束索引：步长] ，这里的意思是从字符串的索引为1的字符开始截取，中间隔一个取，到索引为4的时候结束。
def sortEvenOdd(self, nums: List[int]) -> List[int]:
    nums[::2] = sorted(nums[::2])
    nums[1::2] = sorted(nums[1::2], reverse=True)
    return nums
```

## 2169.得到0的操作数

```python
def countOperations(self, num1: int, num2: int) -> int:
        mx = max(num1,num2)
        mi = min(num1,num2)
        cnt = 0
        while mi !=0:
            mx -= mi
            if mx < mi:
                mx,mi = mi,mx 
            cnt += 1
        return cnt
```

## 2176.统计数组中相等且可以被整除的数对

```python
def countPairs(self, nums: List[int], k: int) -> int:
    num_map = defaultdict(list)
    for i in range(len(nums)):
        num_map[nums[i]].append(i)
    cnt = 0
    for key,idxs in num_map.items():
        for x1,x2 in combinations(idxs,2):
            if x1*x2 % k == 0:
                cnt += 1
    return cnt
```

## 2180.统计各位数字之和为偶数的整数个数

```python
def countEven(self, num: int) -> int:
    cnt = 0
    def digsum(n):
        c = 0
        while n:
            c += n%10
            n//=10
        return c
    for i in range(1,num+1):
        if not digsum(i) %2:
            cnt += 1
    return cnt
```

## 2185.统计包含给定前缀的字符串

```python
def prefixCount(self, words: List[str], pref: str) -> int:
        cnt = 0
        le = len(pref)
        for i in words:
            if i[:le] == pref:
                cnt += 1
        return cnt
```

## 2190.数组中紧跟key之后出现最频繁的数字

```python
def mostFrequent(self, nums: List[int], key: int) -> int:
    nlen = len(nums)
    dic = dict()
    for i in range(nlen-1):
        if nums[i] == key:
            dic[nums[i+1]] = dic.get(nums[i+1],0) + 1
    maxCount = max(dic.values())
    for key,val in dic.items():
        if val == maxCount:
            return key
    return 0
```

## 2194.Excel表中某个范围内的单元格

```python
def cellsInRange(self, s: str) -> List[str]:
    t1,t2 = s.split(":")[0],s.split(":")[1]
    ans = []
    for i in range(ord(t1[0]), ord(t2[0]) + 1):
        for j in range(int(t1[1]), int(t2[1])+1):
            ans.append(chr(i)+str(j))
    return ans
```

## 2200.找出数组中的所有K近邻下标

```python
#解法1
def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
    keys = [i for i,v in enumerate(nums) if v == key]
    return sorted({j for i in keys for j in range(len(nums)) if abs(i - j) <= k})

#解法2
def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
    le = len(nums)
    ans = []
    for i in range(le):
        if nums[i] == key:
            for j in range(max(0,i-k), min(le,i+k+1)):
                ans.append(j)
    return list(set(ans))
```

## 2206.将数组划分成相等数对

```python
def divideArray(self, nums: List[int]) -> bool:
    return sum(val % 2 > 0  for key,val in Counter(nums).items()) == 0
```

## 2210.统计数组中峰和谷的数量

```python
def countHillValley(self, nums: List[int]) -> int:
    num =[nums[0]]
    ans = 0
    for i in nums:
        if i != num[-1]:
            num.append(i)
    le = len(num)
    for i in range(1,le-1):
        if (num[i] > num[i+1] and num[i] > num[i-1]) or (num[i] < num[i+1] and num[i] < num[i-1]):
            ans += 1
    return ans
```

## 2215.找出两数组的不同

```python
#a.differnce(b) 做a与b的差集，返回只在a集合存在的元素
def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
    s1 = set(nums1)
    s2 = set(nums2)
    return [list(s1.difference(s2)),list(s2.difference(s1))]
```

## 2220.转换数字的最少位翻转次数

```python
def minBitFlips(self, start: int, goal: int) -> int:
    return bin(start^goal).count("1")
```

## 2224.转化时间需要的最小操作数

```python
def convertTime(self, current: str, correct: str) -> int:
    z1 = current.split(":")
    z2 = correct.split(":")
    cur1,cur2 =int(z1[0]),int(z1[1])
    cor1,cor2 =int(z2[0]),int(z2[1])
    
    t = cor1*60+cor2 -cur1*60-cur2 
    mi = [60,15,5,1]
    ans = 0
    while t:
        for i in mi:
            if t >= i:
                t -= i
                ans += 1
                break
    return ans
```

## 2231.按奇偶性交换后的最大数字

```python
def largestInteger(self, num: int) -> int:
    sn, book = list(map(int, str(num))), [[], []]
    for x in sn: book[x & 1].append(x)
    for i in range(2): book[i].sort()
    return int(''.join(str(book[x & 1].pop()) for x in sn))
```

## 2235.两整数相加

```python
def sum(self, num1: int, num2: int) -> int:
    return num1 + num2
```

## 2236.判断根节点是否等于子结点之和

```python
def checkTree(self, root: Optional[TreeNode]) -> bool:
    return root.val == root.left.val + root.right.val
```

## 2239.找到最接近0的数字

```python
def findClosestNumber(self, nums: List[int]) -> int:
    mi = float("inf")
    ans = mi
    for i in nums:
        if abs(i) < mi:
            mi = abs(i)
            ans = i
        elif abs(i) == mi and i > ans:
            ans = i
    return ans
```

## 2243.计算字符串的数字和

```python
def digitSum(self, s: str, k: int) -> str:
    def cnt(n):
        ans = 0
        while n:
            ans+=n%10
            n//=10
        return ans
    while len(s) > k:
        tp =""
        le = len(s)
        for i in range(0,le,k):
            tp += str(cnt(int(s[i:i+k])))
        s = tp
    return s
```

## 2248.多个数组求交集

```python
def intersection(self, nums: List[List[int]]) -> List[int]:
    return sorted(list(reduce(lambda x,y:set(x).intersection(set(y)), nums)))
```

## 2255.统计是给定字符串前缀的字符串数目

```python
def countPrefixes(self, words: List[str], s: str) -> int:
    ans = 0
    for i in words:
        le = len(i)
        if s[:le]  == i:
            ans += 1
    return ans
```

## 2259.移除指定数字得到的最大结果

```python
def removeDigit(self, number: str, digit: str) -> str:
    ind =  number.find(digit,0)
    ans = 0
    while ind != -1:
        tp = number[:ind] +number[ind+1:]
        if int(tp) > ans:
            ans = int(tp)
        ind = number.find(digit,ind+1)
    return str(ans)
```

## 2264.字符串中最大的3位相同数字

```python
def largestGoodInteger(self, num: str) -> str:
    ans = -1
    le = len(num)
    for x in range(le-2):
        if num[x] == num[x+1] == num[x+2] and int(num[x]) > ans:
            ans = int(num[x])
    return "" if ans == -1 else str(ans)*3
```

## 2269.找到一个数字的K美丽值

```python
def divisorSubstrings(self, num: int, k: int) -> int:
        sn = str(num)
        le = len(sn)
        ans = 0
        for i in range(le - k + 1):
            t = int(sn[i:i+k])
            if t!=0 and not num % t:
                ans += 1
        return ans
```

## 2273.移除字母异位词后的结果数组

```python
def removeAnagrams(self, words: List[str]) -> List[str]:
    return [w for i, w in enumerate(words) if i == 0 or sorted(words[i - 1]) != sorted(words[i])]
```

## 2278.字母在字符串中的百分比

```python
def percentageLetter(self, s: str, letter: str) -> int:
    ct =Counter(s)
    return ct[letter]*100//sum(ct.values())
```

## 2283.判断一个数的数字计数是否等于数位的值

```python
def digitCount(self, num: str) -> bool:
    mp = Counter(num)
    return all([mp[str(i)] == int(num[i]) for i in range(len(num))])
```

## 2287.重排字符串形成目标字符串

```python
def rearrangeCharacters(self, s: str, target: str) -> int:
    cs = Counter(s)
    bt = Counter(target)
    mi = float("inf")
    for i in target:
        if cs[i]/bt[i] < mi:
            mi = cs[i]/bt[i]
    return int(mi)
```

## 2293.极大极小游戏

```python
def minMaxGame(self, nums: List[int]) -> int:
    while len(nums) != 1:
        tmp_num = []
        for i in range(len(nums)//2):
            if i % 2 == 0:
                tmp_num.append(min(nums[2*i], nums[2*i + 1]))
            else:
                tmp_num.append(max(nums[2*i], nums[2*i + 1]))
        nums = tmp_num     
    return nums[0]
```

## 2299.强密码检测器II

```python
def strongPasswordCheckerII(self, password: str) -> bool:
    st = "!@#$%^&*()-+"
    ans = [0]*5
    le = len(password)
    if le >= 8: ans[0] = 1
    for t in range(le):
        i = password[t]
        if i.isdigit():
            ans[1] = 1
        elif i.islower():
            ans[2] = 1
        elif i.isupper():
            ans[3] = 1
        elif i in st:
            ans[4] = 1
        if t!=le-1 and password[t] == password[t+1]:
            return False
    return  reduce(lambda x,y: x*y, ans) != 0
```

## 2303.计算应缴税款总额

```python
def calculateTax(self, brackets: List[List[int]], income: int) -> float:
    res=0
    prev=0
    for u,p in brackets:
        if income>u:
            res+=(u-prev)*p/100
        else:
            res+=(income-prev)*p/100
            return res
        prev=u
    return res
```

## 2309.兼具大小写的最好英文字母

```python
def greatestLetter(self, s: str) -> str:
    for i in range(122,96,-1):
        c = chr(i) 
        if c in s and c.upper() in s:
            return c.upper()
    return ""
```

## 2315.统计星号

```python
def countAsterisks(self, s: str) -> int:
    k = s.split("|")
    le = len(k)
    ans = 0
    for i in range(le):
        if not i%2:
            ans += k[i].count("*")
    return ans
```

## 2319.判断矩阵是否是一个X矩阵

```python
def checkXMatrix(self, grid: List[List[int]]) -> bool:
    return all((v != 0) == (i == j or i + j == len(grid) - 1) for i, row in enumerate(grid) for j, v in enumerate(row))
```

## 2325.解密消息

```python
def decodeMessage(self, key: str, message: str) -> str:
    rec = {}
    key = key.replace(" ","")
    ind = 0
    for i in key:
        if i not in rec.keys():
            rec[i] = ind
            ind += 1
    ans = ""
    for i in message:
        if i != " ":
            ans += chr(rec[i] + 97)
        else:
            ans += " "
    return ans
```

## 2331.计算布尔二叉树的值

```python
def evaluateTree(self, root: Optional[TreeNode]) -> bool:
    def dfs(root):
        if not root.left and not root.right:
            return root.val
        if root.val==2:
            return dfs(root.left) or dfs(root.right)
        return dfs(root.left) and dfs(root.right)
    if dfs(root)==1:
        return True
    return False
```

## 2335.装满被子需要的最短总时长

```python
#ceil函数返回向上取整
def fillCups(self, amount: List[int]) -> int:
    return max(max(amount), math.ceil(sum(amount)/2))
```

## 2341.数组能形成多少数对

```python
def numberOfPairs(self, nums: List[int]) -> List[int]:
    ct = Counter(nums)
    a1 = 0
    a2 = 0
    for i in ct.items():
        a1 += i[1]//2
        a2 += i[1]%2
    return [a1,a2]
```

## 2347.最好的扑克手牌

```python
def bestHand(self, ranks: List[int], suits: List[str]) -> str:
    if len(set(suits)) == 1:
        return "Flush"
    st = len(set(ranks))
    ct= Counter(ranks)
    if st <= 3:
        for i in set(ranks):
            if ct[i] >= 3:
                return "Three of a Kind"
        return "Pair"
    elif st == 4:
        return "Pair"
    return "High Card"
```

## 2351.第一次出现两次的字母

```python
def repeatedCharacter(self, s: str) -> str:
    st= []
    for i in s:
        if i in st:
            return i
        st.append(i)
    return ""
```

# MEDIUM

## 2.两数相加

```python
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    # 初始化链表
    head = tree = ListNode()
    val = tmp = 0
    # 当三者有一个不为空时继续循环
    while tmp or l1 or l2:
        val = tmp
        if l1:
            val = l1.val + val
            l1 = l1.next
        if l2:
            val = l2.val + val
            l2 = l2.next

        tmp = val // 10
        val = val % 10

        # 实现链表的连接
        tree.next = ListNode(val)
        tree = tree.next
    return head.next
```

## 3.无重复字符的最长子串

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    tmp = ""
    ans = 0
    for t in s:
        if t not in tmp:
            tmp += t
        else:
            tmp = tmp[tmp.index(t)+1:]
            tmp += t
        ans = max(ans,len(tmp))
    return ans
```

## 5.最长回文子串

```python
#中心扩展法
def expandAroundCenter(self, s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - 1

def longestPalindrome(self, s: str) -> str:
    start, end = 0, 0
    for i in range(len(s)):
        left1, right1 = self.expandAroundCenter(s, i, i)
        left2, right2 = self.expandAroundCenter(s, i, i + 1)
        if right1 - left1 > end - start:
            start, end = left1, right1
        if right2 - left2 > end - start:
            start, end = left2, right2
    return s[start: end + 1]
```

## 6.Z字形变换

```python
#巧用flag
def convert(self, s: str, numRows: int) -> str:
    if numRows < 2: 
        return s
    res = ["" for _ in range(numRows)]
    i, flag = 0, -1
    for c in s:
        res[i] += c
        if i == 0 or i == numRows - 1: flag = -flag
        i += flag
    return "".join(res)
```

## 7.整数反转

```python
def reverse(self, x: int) -> int:
    flag = False
    if x < 0:
        flag = True
        x = -x
    x = int(str(x)[::-1])if not flag else -int(str(x)[::-1])
    return x if -2147483648 < x < 2147483647 else 0
```

# HARD