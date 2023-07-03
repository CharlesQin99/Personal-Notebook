# JAVA

## java教学

![](pic\1.png)

![](pic\2.png)

![](pic\3.png)

![](pic\4.png)

![](pic\5.png)

![](pic\6.png)

解释型语言是按行解释翻译，c这样的语言是整体翻译，java是混合型的（java不是运行在操作系统的，而是运行在java提供好的虚拟机上的）

![](pic\7.png)

# EASY

## 1.两数之和

Map<Integer, Integer> map = new HashMap<>()

- map.containsKey
- map.put
- map.get

```java
由于哈希查找的时间复杂度为O(1)，所以可以利用哈希容器 map 降低时间复杂度
遍历数组 nums，i 为当前下标，每个值都判断map中是否存在 target-nums[i] 的 key 值
如果存在则找到了两个值，如果不存在则将当前的 (nums[i],i) 存入 map 中，继续遍历直到找到为止
如果最终都没有结果则抛出异常

class Solution {
    public int[] twoSum(int[] nums, int target) {
    	Map<Integer,Integer> map = new HashMap<>();
    	for (int i =0;i<nums.length;i++)
    	{
    		if(map.containsKey(target - nums[i]))
    		{
    			return new int[] {map.get(target - nums[i]), i};
    		}
    		map.put(nums[i],i);  //数组和下标的键值对
    	}
        throw new IllegalArgumentException("No answer")
    }
}
```

## 9.回文数

*字符串翻转的方法*

String reversedStr = (new StringBuilder(x + "")).reverse().toString();

**JAVA 中int类型转String类型的三种通常方法：**

- Integer.toString(int i)
- String.valueOf(int i)
- i + “”; //i 为 int类型，int+string型就是先将int型的i转为string然后跟上后面的空string。

三种方法效率排序为：

**Integer.toString(int i) > String.valueOf(int i) > i+""**

*字符串比较*

- **equals()** 方法：将逐个地比较两个字符串的每个字符是否相同。如果两个字符串具有相同的字符和长度，它返回 true，否则返回 false。
- **equalsIgnoreCase()** 方法：作用和语法与 equals() 方法完全相同，唯一不同的是 equalsIgnoreCase() 比较时不区分大小写
- **compareTo()** 方法：用于按字典顺序比较两个字符串的大小，该比较是基于字符串各个字符的 Unicode 值

> str = A
>
> str1 = a
>
> str.compareTo(str1)的结果是：-32
>
> str1.compareTo(str)的结果是：32
>
> str1.compareTo('a')的结果是：0

```java
class Solution {
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        String reversedstr = (new StringBuilder(Integer.toString(x))).reverse().toString();
        return reversedstr.equals(Integer.toString(x));
    }
}
```

## 13.罗马数字转整数

*字符串截取*

s.substring(i, i+1)

s.charAt(i)

```java
class Solution {
    public int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("V", 5);
        map.put("X", 10);
        map.put("L", 50);
        map.put("C", 100);
        map.put("D", 500);
        map.put("M", 1000);
        int len  = s.length();
        if(len == 0) return 0;
    
        int ans = 0;
        for(int i = 0;i < len - 1; i++)
        {
            if(map.get(s.substring(i, i+1)) < map.get(s.substring(i+1, i+2)))
            {
                ans -= map.get(s.substring(i, i+1));
            }
            else
            {
                ans += map.get(s.substring(i, i+1));
            }
        }
        
        return ans + map.get(s.substring(len-1, len));
    }
}
```

## 14.最长公共前缀

*按字典序排序后对第一个和最后一个进行比较即可*

  Arrays.sort(strs)

> 下列单词就是按照**字典序**进行排列的：
>
> as
>
> aster
>
> astrolabe
>
> astronomy
>
> astrophysics
>
> at
>
> ataman
>
> attack
>
> baa

```java
class Solution {
    //先排序后判断
    public String longestCommonPrefix(String[] strs) {
        //特殊情况过滤：当数组只有一个字符时,返回该字符串
        if(strs.length == 1)
        {
            return strs[0];
        }
        Arrays.sort(strs);
        int index = 0;
        int len = Math.min(strs[0].length(),strs[strs.length-1].length());
        for(int i = 0; i<len;i++)
        {
            if(strs[0].charAt(i) != strs[strs.length-1].charAt(i))
            {
                break;
            }
            index++;
        }
        return strs[0].substring(0,index);
    }
}
```

*利用indexOf(prefix)来判断公共前缀*

indexOf：返回特定**子字符串第一次在源字符串中的位置**。如果源字符中不存在目标字符，则返回-1。

```java
cla、ss Solution {
    public String longestCommonPrefix(String[] strs) {
        String prefix = strs[0];//默认第一个单词为最长公共前缀
        for (int i = 1; i < strs.length; i++) 
        {
            while (strs[i].indexOf(prefix) !=0) {	//只有indexOf=0,才说明是公共前缀
                prefix = prefix.substring(0, prefix.length() - 1);//每次去除末尾的字符
                if (prefix.isEmpty()) 
                {
                    return "";
                }
            }
        }
        return prefix;
    }
}
```

## 20.有效的括号

**LinkedList**是实现了List和Deque接口的**双向链表**，实现了所有可选列表的操作，允许存放所有元素（包括null）

```
常见用法：
public LinkedList()
public LinkedList(Collection<? extends E> c)

public boolean contains(Object o)  判断是否包含
public int size()  返回集合大小

public E getFirst()  获取第一个元素
public E getLast()  获取最后一个元素
public E get(int index)  根据索引获取元素
public E set(int index, E element)  在指定索引处添加一个元素（覆盖原数据）
public int indexOf(Object o)  返回指定元素索引
public int lastIndexOf(Object o)  返回指定元素出现的最后一个索引

public boolean add(E e)  添加一个元素
public void add(int index, E element)  在指定位置插入元素
public void addFirst(E e)  在集合前部增加一个元素
public void addLast(E e)  在集合尾部增加一个元素
public boolean addAll(Collection<? extends E> c)  把另一个集合全部添加到当前集合
public boolean addAll(int index, Collection<? extends E> c) 在指定位置把另一个集合全部添加到当前集合

public boolean remove(Object o)  根据指定对象删除元素
public E remove(int index)  根据指定索引删除元素
public E removeFirst()  删除第一个元素
public E removeLast()  删除最后一个元素

```

***字符串遍历 for (Character c: s.toCharArray())***

```java
class Solution {
	private static final Map<Character,Character> map = new HashMap<Character, Character>(){{
        put('[',']');put('{','}');put('(',')');put('?','?');
    }};
    
    //?放在栈底就是防止第一个字符是右括号，此时stack为空时pop报错
    public boolean isValid(String s){
        if(s.length == 0 || !map.containsKey(s.charAt(0))) return false;
        LinkedList<Character> stack  = new LinkedList<Character>(){{ add("?"); }};
        for (Character c: s.toCharArray())
        {
            //也可以加一句判断是否为？，不过最后的size == 1也会筛选出这种错误
            if(map.containsKey(c)) stack.addLast(c);
            else if(map.get(stack.removeLast())!=c) return false;
        }
        return stack.size() == 1;
    }
}
```

## 21.合并两个有序列表

递归两大要素：

- **终止条件**
- **如何递归**

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        else if (l2 == null) {
            return l1;
        }
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }

    }
}
```

## 26.删除有序数组中的重复项

***双指针***

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int left = 0;
        int right = 1;
        while(right < nums.length)
        {
            if(nums[left] != nums[right])
            {
                if(right - left > 1)  //优化部分，如果没有重复元素直接移动指针，不用复制一遍
                {
                    nums[left + 1] = nums[right];
                }
                left++;
            }
            right++;
        }
        return  left + 1;
    }
}
```

## 27.移除元素

双指针

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int left = 0;
        for (int right = 0; right < n; right++) 
        {
            if (nums[right] != val) 
            {
                nums[left] = nums[right];
                left++;
            }
        }
        return left;
    }
}
```

双指针优化: **两个指针初始时分别位于数组的首尾，向中间移动遍历该序列**

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int left = 0;
        int right = nums.length;
        while (left < right) 
        {
            if (nums[left] == val) 
            {
                nums[left] = nums[right - 1];
                right--;
            }else{
                left++;
            }
        }
        return left;
    }
}
```

## 35.搜索插入位置

***二分查找的写法***

```
class Solution {
    public int searchInsert(int[] nums, int target) {
        int ans = 0;
        for(int i = 0;i < nums.length;i++)
        {
            if(target > nums[i])
            {
                ans++;
            }
            else
            {
                return ans;
            }
        }
        return ans;
    }
}

    //闭区间写法
    // private int lowerBound(int[] nums, int target) {
    //     int left = 0, right = nums.length - 1; // 闭区间 [left, right]
    //     while (left <= right) { // 区间不为空
    //         // 循环不变量：
    //         // nums[left-1] < target
    //         // nums[right+1] >= target
    //         int mid = left + (right - left) / 2;
    //         if (nums[mid] < target)
    //             left = mid + 1; // 范围缩小到 [mid+1, right]
    //         else
    //             right = mid - 1; // 范围缩小到 [left, mid-1]
    //     }
    //     return left; // 或者 right+1
    // }
```

## 58.最后一个单词的长度

trim():去除首尾空格

split():注意不能为空

```java
pulic int lengthOfLastWord(String s){
	s = s.trim(); //去除首尾空格
	String[] arr =s.split(" ");
	return arr[arr.length - 1].length();
}
```

## 66.加一

```java
 public int[] plusOne(int[] digits) {
        int len = digits.length;
        for(int i = len-1;i>=0;i--)
        {
            digits[i] = (digits[i] + 1)%10;
            if(digits[i] != 0)
            {
                return digits;
            }
        }
        //全9的情况
        digits = new int [len + 1];
        digits[0] = 1;
        return digits;
    }
```

## 67.二进制求和

Integer.**toBinaryString**

Integer.**parseInt**

 Integer.**toString**

```java
//利用java自带的高精度运算，先转为十进制，然后用二进制相加
public String addBinary(String a, String b) {  // 将数字转换成二进制
    return Integer.toBinaryString(
        Integer.parseInt(a, 2) + Integer.parseInt(b, 2)
    );
}

public String addBinary(String a, String b) {
    StringBuilder ans = new StringBuilder();    //答案
    int i = a.length() - 1, j = b.length() - 1; //从最后开始遍历（个位）
    int t = 0;  //进位
    while(i >= 0 || j >= 0 || t != 0) { //如果没有遍历完两个数，或者还有进位
        if(i >= 0) t += a.charAt(i--) - '0';    //如果a还有数
        if(j >= 0) t += b.charAt(j--) - '0';    //如果b还有数
        ans.append(t % 2);  //加入当前位的和取模
        t /= 2; //进位
    }
    return ans.reverse().toString();    //由于计算之后是个位在第一位，所以要反转
}
```

一、stringbuffer和stringbuilder的区别

1.线程安全

StringBuffer：线程安全

StringBuilder：线程不安全。

因为 StringBuffer 的所有公开方法都是 synchronized 修饰的，而 StringBuilder 并没有 synchronized 修饰。

2.缓冲区

**StringBuffer 每次获取 toString 都会直接使用缓存区的 toStringCache 值来构造一个字符串。**

**StringBuilder 则每次都需要复制一次字符数组，再构造一个字符串。**

所以， StringBuffer 对缓存区优化，不过 StringBuffer 的这个toString 方法仍然是同步的。

3.性能

既然 StringBuffer 是线程安全的，它的所有公开方法都是同步的，StringBuilder 是没有对方法加锁同步的，所以毫无疑问，**<u>StringBuilder 的性能要远大于 StringBuffer。</u>**

二、StringBuffer的常用方法

StringBuffer类中的方法主要偏重于对于字符串的变化，例如追加、插入和删除等，这个也是StringBuffer和String类的主要区别。

## 69.x的平方根

**牛顿迭代法**：
$$
X_{k+1} = X_{k} - \frac{f(X_k)}{f^{'}(X_k)}
$$
函数上任一点(x,f(x))处的切线斜率是2x。那么，x-f(x)/(2x)就是一个比x更接近的近似值。代入 f(x)=x^2-a得到x-(x^2-a)/(2x)，也就是**(x+a/x)/2**。

```java
class Solution {
    public int mySqrt(int x) {
        if(x==0) return 0;
        return (int)(sqrts(x, x));
      }
    
    public double sqrts(double x, int res){
        double xk = (x + res / x) / 2;
        if (xk == x)	//收敛
        {
            return x;
        } else {
            return sqrts(xk, res);
        }
    } 
}
```

## 70.爬楼梯

**动态规划**，注意回溯法的话时间复杂度太高

```java
public int climbStairs(int n) {
    if(n <= 2)  return n;
    int[] dp = new int[n+5];
    dp[2] = 2;
    dp[1] = 1;
    for(int i = 3;i <= n;i++)
    {
        dp[i] = dp[i-2] + dp[i-1];
    }
    return dp[n];
}

//解法2 数学解法
 public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int step1 = 1;
        int step2 = 2;
        int cur = step1 + step2;
        for (int i = 2; i < n; ++i) {
            cur = step1 + step2;
            step1 = step2;
            step2 = cur;
        }
        return cur;
    }
```

## 83.删除排序链表中的重复元素

```java
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) return head
        ListNode cur = head;
        while(cur.next != null)
        {
        	if(cur.val == cur.next.val)
        		cur.next = cur.next.next;
        	else
        		cur = cur.next;
        }
        return head;
    }
```

## 88.合并两个有序数组

**Arrays.sort()**:进行数组排序

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    for (int i = 0; i != n; ++i) 
    {
        nums1[m + i] = nums2[i];
    }
    Arrays.sort(nums1);
}

//逆向双指针
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int p1 = m-1,p2 = n-1,tail = m + n - 1;
    while(tail >= 0)
    {
        int cur;
        if(p1 == -1 ){
            cur = nums2[p2--];
        }else if(p2 == -1){
            cur= nums1[p1--];
        }else if (nums1[p1] > nums2[p2]) {
            cur = nums1[p1--];
        } else {
            cur = nums2[p2--];
        }
        nums1[tail--] = cur;
    }
}
```

## **94.二叉树的中序遍历**

- 先：根，左，右
- 中：左，根，右
- 后：左，右，根

**先序：访问到一个节点后，即刻输出该节点的值，并继续遍历其左右子树。(根左右)**
**中序：访问到一个节点后，将其暂存，遍历完左子树后，再输出该节点的值，然后遍历右子树。(左根右)**
**后序：访问到一个节点后，将其暂存，遍历完左右子树后，再输出该节点的值。(左右根)**

### 递归

简单**dfs**

时间复杂度：O*(*n)
空间复杂度：O*(*h)，ℎ是树的高度

递归的调用过程是**不断往左边走，当左边走不下去了，就打印节点，并转向右边，然后右边继续这个过程。**
**我们在迭代实现时，就可以用栈来模拟上面的调用过程**。

```java
public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		dfs(res, root);
		return res;
	}
	
	void dfs(List<Integer> res, TreeNode root) {
		if(root==null) return;
		dfs(res, root.left);
		res.add(root.val);
		dfs(res, root.right);
	}
```

### 栈模拟递归

时间复杂度：O*(*n)
空间复杂度：O*(*h)，ℎ是树的高度

```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<Integer>();
    Stack<TreeNode> stack = new Stack<TreeNode>();
    while(stack.size()>0 || root!=null) {
        //不断往左子树方向走，每走一次就将当前节点保存到栈中
        //这是模拟递归的调用
        if(root!=null) {
            stack.add(root);
            root = root.left;
            //当前节点为空，说明左边走到头了，从栈中弹出节点并保存
            //然后转向右边节点，继续上面整个过程
        } else {
            TreeNode tmp = stack.pop();
            res.add(tmp.val);
            root = tmp.right;
        }
    }
    return res;
}
```

### 莫里斯算法

morris 算法

***把一棵二叉树改成一段链表结构***

用递归和迭代的方式都使用了辅助的空间，而**<u>莫里斯遍历的优点是没有使用任何辅助空间。</u>**
缺点是改变了整个树的结构，强行把一棵二叉树改成一段链表结构。

```java
public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		TreeNode pre = null;
		while(root!=null) {
			//如果左节点不为空，就将当前节点连带右子树全部挂到
			//左节点的最右子树下面
			if(root.left!=null) {
				pre = root.left;
				while(pre.right!=null) {
					pre = pre.right;
				}
				pre.right = root;
				//将root指向root的left
				TreeNode tmp = root;
				root = root.left;
				tmp.left = null;
			//左子树为空，则打印这个节点，并向右边遍历	
			} else {
				res.add(root.val);
				root = root.right;
			}
		}
		return res;
	}
```

## 100.相同的树

***&&比&效率更高***

**&**运算符，当两边都返回true时，按位与才返回true。
**&&**操作符第一个表达式为 false时，结果为 false，并且不再计算第二个表达式。

> **||和&&左边有一个不通过就不去执行右边了，不会浪费时间。**
>
> **|和&不管左边有没有通过，都要两边先执行一遍，效率低**

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if(p == null && q==null) return true;
    if(q == null || p == null || p.val != q.val)
    {
        return false;
    }
    return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);	
}
```



# MEDIUM

# HARD

# 