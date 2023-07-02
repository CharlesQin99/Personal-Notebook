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



# MEDIUM

# HARD

# 