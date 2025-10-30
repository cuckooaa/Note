## C++ 快速入门与刷题重点（与 C/Java/Python 对比）

### 1\. 编译和内存管理

| 特性 | C | Java | Python | C++ (重点) | 对比/刷题提示 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **类型** | 结构化/过程式 | 纯面向对象（一切皆对象） | 动态类型 | **混合式**（支持面向对象，也支持过程式） | C++ 是 C 的超集，兼容 C 的语法和过程式编程。 |
| **编译** | 编译成机器码 | 编译成字节码（JVM解释执行） | 解释执行 | **编译成机器码** (与 C 类似) | LeetCode 平台通常使用 $\texttt{g++}$ 编译，与 C 编译过程相似。 |
| **内存管理** | 手动 ( $\texttt{malloc}$ / $\texttt{free}$ ) | **自动** (垃圾回收 GC) | **自动** (垃圾回收 GC) | **手动** ( $\texttt{new}$ / $\texttt{delete}$ ) + **自动** (RAII, 智能指针) | 刷题中，**栈**分配和 **$\texttt{new}$** 是主要的，但现代 C++ 鼓励使用 **智能指针** ($\texttt{std::shared\_ptr}$ / $\texttt{std::unique\_ptr}$)，尤其在复杂数据结构中。 |

-----

### 2\. 标准库与容器（刷题核心）

C++ 的强大很大程度上来自于它的**标准模板库 (STL)**，这在刷 LeetCode 时是效率的保证。

| C++ STL 容器 | 类似 C 概念 | 类似 Java 概念 | 类似 Python 概念 | 刷题常用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **$\texttt{std::vector<T>}$** | 动态数组 (无) | $\texttt{ArrayList<T>}$ | $\texttt{list}$ | 动态数组、栈、序列存储。**最常用容器**。 |
| **$\texttt{std::string}$** | $\texttt{char*}$ 字符串 | $\texttt{String}$ | $\texttt{str}$ | 字符串操作。 |
| **$\texttt{std::list<T>}$** | 双向链表 (自己实现) | $\texttt{LinkedList<T>}$ | $\texttt{deque}$ | 双向链表，但 LeetCode 中通常直接用指针实现链表题。 |
| **$\texttt{std::map<K, V>}$** | (无) | $\texttt{TreeMap<K, V>}$ | (无完全对应，字典默认是哈希) | 基于红黑树的有序映射（字典），用于需要**排序**的 K-V 存储。 |
| **$\texttt{std::unordered\_map<K, V>}$** | (无) | $\texttt{HashMap<K, V>}$ | $\texttt{dict}$ | 基于哈希表的无序映射，用于快速查找（**最常用**）。 |
| **$\texttt{std::set<T>}$** | (无) | $\texttt{TreeSet<T>}$ | (无完全对应，集合默认是哈希) | 有序集合，用于去重和查找，**有排序需求**。 |
| **$\texttt{std::unordered\_set<T>}$** | (无) | $\texttt{HashSet<T>}$ | $\texttt{set}$ | 无序集合，用于快速去重和查找（**最常用**）。 |
| **$\texttt{std::queue<T>}$** | (无) | $\texttt{Queue<T>}$ | $\texttt{queue}$ ( $\texttt{collections.deque}$ ) | 广度优先搜索 (BFS)。 |
| **$\texttt{std::stack<T>}$** | (无) | $\texttt{Stack<T>}$ | $\texttt{stack}$ ( $\texttt{list}$ 或 $\texttt{collections.deque}$ ) | 栈操作，深度优先搜索 (DFS) 的辅助结构。 |
| **$\texttt{std::priority\_queue<T>}$** | (无) | $\texttt{PriorityQueue<T>}$ | $\texttt{heapq}$ | 堆（默认大顶堆），用于 Top K 问题、Dijkstra 等。 |

**STL 刷题提示：**

  * **头文件：** 必须 $\texttt{\#include <vector>}$ 等，不能 $\texttt{\#include <iostream>}$ 包含一切（虽然某些 LeetCode 编译器可以）。
  * **命名空间：** 使用 $\texttt{using namespace std;}$ 可以避免每次都写 $\texttt{std::}$。

-----

### 3\. C++ 语言特性重点

#### A. 输入/输出 (I/O)

  * **C++ 风格：** $\texttt{cout}$ 和 $\texttt{cin}$
    ```cpp
    #include <iostream>
    using namespace std;

    int num;
    cin >> num;
    cout << "Number is: " << num << endl;
    ```
    *与 C 的 $\texttt{scanf}$ / $\texttt{printf}$ 相比，类型安全，更面向对象；与 Java / Python 相比，语法更简洁。*

#### B. 指针 vs 引用 (Reference)

  * **C 的指针 ( $\texttt{*}$ )：** 存储地址，需要解引用，可以指向空 ( $\texttt{NULL}$ / $\texttt{nullptr}$ )。
  * **Java/Python 的引用：** 变量名即是对对象的引用，没有显式的指针操作。
  * **C++ 的引用 ( $\texttt{\&}$ )：** 是变量的别名，一旦初始化不能改变指向，**不能指向空**。

**刷题重点：**

1.  **传参：** C++ 中，**引用传参** ($\texttt{void func(int\& num)}$) 是一个重要概念，它允许你在函数内修改外部变量，且**避免了像 C 一样的值拷贝**，效率高。
2.  **$\texttt{const}$ 引用：** 用于**高效地传入大型对象**（如 $\texttt{vector}$ 或 $\texttt{string}$），同时确保函数内部不会修改它。 ($\texttt{void func(const vector<int>\& v)}$)

#### C. $\texttt{const}$ 关键字

C++ 中 $\texttt{const}$ 的使用更加广泛和严格：

  * **常量：** $\texttt{const int MAX = 10;}$
  * **修饰指针/引用：** 确保不能通过该指针/引用修改值。
  * **修饰成员函数 (面向对象)：** 确保该函数不会修改对象的**成员变量**。

#### D. 面向对象 (OOP) - 简化版

虽然 LeetCode 很多时候不需要完整的面向对象，但您需要知道**类 ( $\texttt{class}$ )** 的基本用法，因为链表和树节点就是用它定义的。

```cpp
// 链表节点定义 (ListNode 是 LeetCode 常用结构)
struct ListNode { // struct 类似于 class，但默认 public
    int val;
    ListNode *next; // 指针！与 Java/Python 的引用不同

    // 构造函数 (与 Java 类似，与 Python 的 __init__ 类似)
    ListNode() : val(0), next(nullptr) {} // 默认构造
    ListNode(int x) : val(x), next(nullptr) {} // 带参构造
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
```

  * **注意：** $\texttt{next}$ 是一个指针 ($\texttt{*}$)，这意味着它存储的是下一个节点的内存地址，而不是下一个节点本身。这是 C++ 与 Java/Python 的主要区别之一。

-----

### 4\. 刷题必备的 C++ 技巧

| 技巧 | 示例代码 | 作用 |
| :--- | :--- | :--- |
| **迭代器 (Iterator)** | $\texttt{for (auto it = v.begin(); it != v.end(); ++it)}$ | 遍历 STL 容器，类似于 Java 的迭代器。 |
| **Range-based $\texttt{for}$ 循环** | $\texttt{for (int\& num : vec) \{ num *= 2; \}}$ | 类似 Python 的 $\texttt{for item in list:}$，如果使用 $\texttt{\&}$ 可以修改元素。 |
| **$\texttt{auto}$ 关键字** | $\texttt{auto it = map.begin();}$ $\texttt{auto len = s.length();}$ | 自动推断变量类型，使代码更简洁 (类似 Python 的动态类型，但 C++ 仍然是静态类型)。 |
| **$\texttt{std::sort}$** | $\texttt{std::sort(v.begin(), v.end());}$ | **对 $\texttt{vector}$ 排序**，效率高。可配合 $\texttt{lambda}$ 表达式自定义排序规则。 |
| **Lambda 表达式** | $\texttt{[](int a, int b) \{ return a < b; \}}$ | 简洁地定义匿名函数，常用于 $\texttt{sort}$ 或 $\texttt{priority\_queue}$ 的自定义比较器。 |

### 总结

您从 C 基础可以直接过渡到 C++ 的**指针**、**内存管理**和**过程式编程**部分。

您从 Java/Python 基础可以直接理解 C++ 的**面向对象**概念、**容器 (STL)** 的作用以及**函数**的结构。

**建议您从 $\texttt{std::vector}$、$\texttt{std::unordered\_map}$、$\texttt{std::sort}$ 和引用传参开始练习，这些是 LeetCode 中最核心、最常用的 C++ 元素。**
