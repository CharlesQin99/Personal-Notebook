# Web3

**Web1**

**静态内容，信息聚合，只读**，如早期的Yahoo

**Web2**

**可交互**，读写，社交媒体，**数据存储在中心化服务器**，例如twitter

**Web3**

使用**区块链技术，智能合约，去中心化**，例如以太坊





## Web3基础

***密码学是区块链底层的安全保障***

- 非对称加密：公钥密码学
- 哈希函数

**自己持有一个私钥，用来签名交易**，然后区块链用公钥来解密来证明是本人做的交易

![](pic\0.png)

**比特币**

2008年11月1日由中本聪发明

**去中心化的电子记账系统**，Pow共识机制

**以太坊**

2013年由Vitalik（V神）提出，**“世界计算机”**，***去中心化的全球计算机，智能合约***

22年由Pow转为权益证明PoS共识机制

***结构比比特币复杂很多***

![](pic\1.png)

**智能合约**

以太坊上运行的一个**程序** ，**代码即法律，部署在区块链上，无需第三方机构公证**

![](pic\2.png)

***可以通过WTF-Solidity学习编写智能合约***

由于智能合约开源，不可篡改，很容易遭到黑客攻击

![](pic\3.png)

**加密钱包**

例如 **METAMASK**

开始时会生成**助记词（私钥）**，公钥是**链上地址（用户名），是可以公开的**

使用加密钱包在区块链上管理链上资产，***如果谁拿到了私钥，谁就拥有了你的资产***

**DeFI**

**去中心化金融**，建立在区块链、智能合约平台上的金融类产品工具

![](pic\4.png)

**NFT**

**非同质化代币，是独特的数字代币，不能互换**

链上艺术品/头像，虚拟房产

**DAO**

去中心化自治组织，**通过智能合约保持运转的组织**

![](pic\5.png)

## 密码学基础

***密码学是区块链的底层安全机制的保障***

![](pic\6.png)

### 科尔霍夫原则

对于一个密码学系统，**<u>应当仅有密钥是保密</u>**，其余算法和参数都应该是公开的

### 对称加密

加解密都是使用相同的密钥

古典密码学：

- 凯撒密码
- 维吉尼亚密码

现代对称加密算法：

- **分组加密：AES,DES**
- **流密码：伪随机数生成器**

### **非对称加密**

加解密使用不同的密钥：公钥，私钥

缺点是**效率较低**，一般使用非对称交换密钥，再利用对称加密通信

非对称加密算法：

- **RSA**，基于大整数分解的困难
- **离散对数**，例如**<u>椭圆曲线ECC</u>**

> **比特币就是采用椭圆曲线的数字签名来加解密**

![](pic\7.png)

### 数字签名

***与正常的加解密有对称性***

- 普通：**公钥加密，私钥解密**，没有私钥的话无法获取明文相关消息
- 数字签名：用**<u>自己私钥签名，别人用公钥验证签名有效性</u>**，没有私钥的话就无法伪造签名

![](pic\8.png)

### 哈希算法

就是***消息摘要***，把无穷空间内的消息映射到有限空间内

![](pic\9.png)

## **钱包地址**

***每个钱包地址对应一对公私钥***

***私钥是证明账户的唯一办法，一旦泄露钱包就会被窃取，任何情况不要复制私钥和助记词***

**交易**

![](pic\10.png)

## 挖矿

由于hash算法的好性质，除了***暴力枚举没有别的挖矿方法***

只能不停地试试，消息上加上随机数算哈希看满不满足，直到能出块

> 通过算力的分散性保证去中心化的出块

**<u>每一个块的头部都包含上一个区块的哈希</u>**，如果想要修改之前区块的某个内容，需要从那个块开始整个后面的块都要修改，因此**很难对抗全网算力算出新的链，即篡改区块链信息**

## 默克尔树

依靠哈希来***快速确认某个值是否在一个集合中***，常用于区块存储交易

![](pic\11.png)

## 以太坊

Layer2协议：**缓和以太坊交易的拥堵**，链上处理的交易转移到链下处理

***交易不需要在以太坊进行，只要在线下计算，然后把执行结果和压缩数据上传到主链就可以了***

### Rollup

***一次验证多个交易，把大量交易打包处理，然后压缩后上传主链***

![](pic\12.png)

> <u>**以太坊通过诸如Optimistic Rollups和ZK Rollups等技术实现这些功能，可以减轻主网管理交易的负担，从而实现更大的交易容量和吞吐量（每秒更高的交易量），所有这些都带来了更加无缝和实用的用户体验。以太坊上Layer2的示例包括 Arbitrum、Optimism、Loopring 和 zkSync 等解决方案**</u>

#### **Optimistic Rollups**

**挑战机制**，用**保证金来**做保证，挑战期一般为1到2周

***使用<u>欺诈证明</u>**，如果有人发现错误，**<u>*上交保证金后提交挑战（防止DDOS攻击，无限发起挑战）*</u>**，***之后由智能合约验证，如果确实出现错误，执行该批次的Rollup和之后的交易都会回滚，处罚提交错误的欺诈者的保证金，并用来奖励举报者****

![](pic\13.png)

- 优点：**技术成熟，易于实现**
- 缺点：**挑战期过长**，**<u>因为交易回滚代价很大</u>**，用户体验差,需要***足够的挑战者才能验证Rollup区块的合法性***

**ARBITRUM**

***采用多轮交互式欺诈证明，原理类似二分法，实际是分更多，如200份***

思路：将尽可能多的将验证工作**<u>放到链下执行，降低链上成本</u>**，同时可以让ARBITRUM能***支持复杂智能合约***

![](pic\14.png)

#### ***zk-Rollups***

**zero-knowledge proof**，利用**<u>零知识证明</u>保障交易的安全性**，安全性依赖于密码学原理，**<u>*主要优点在于不需要挑战期*</u>**

使用**<u>有效性证明</u>**，每次提交交易需要提交证明，***当场就可以链上验证，交易时间提高很多，但是所依赖的密码学原理很复杂***

![](pic\15.png)

例：

![](pic\16.png)

**zk-SNARK网络在区块链中的应用：Filecoin，Zcash**

![](pic\17.png)

##### zkSync

采用**zk-SNARK技术**

安全性依赖于**<u>初始化信任设置，参与者必须诚实，不能修改参数</u>**

![](pic\18.png)

支持***链上和链下两种数据存储方案***，但是**<u>链下存储其实是和去中心化理论是背道而驰的</u>**

![](pic\19.png)

##### STARKWARE

基于**zk-STARK技术**，相当于zk-SNARK的升级版本，社区还在起步阶段

- 无需信任设置
- 扩展性
- 抗量子攻击

![](pic\20.png)

**<u>以太坊的智能合约语言Solidity转到STRAK兼容的格式很复</u>**杂，所以**STARK团队开发了Cairo编程语言来作为中间转换**

![](pic\21.png)

![](pic\22.png)

![](pic\23.png)

### 智能合约

***以太坊使得任何人能够创造合约和去中心化应用，并在其中设定自由定义的规则，交易方式等等***

**EVM（以太坊虚拟机）**：以太坊中**<u>智能合约的运行环境</u>**

> 运行在EVM内部的代码不能接触网络，文件系统或其他进程

**哈希算法**

![](pic\24.png)

**区块**：**上一页哈希值 + 本页的交易 + 幸运数字**

前两个属性是固定的，这一个区块满足某个性质，依靠算力，谁先算出来这个区块需要的哈希值，就拥有了这区块。然后他向全网广播，然后由矿工们验证，验证成功后，其他矿工就区块连上去，然后继续寻找。

![](pic\25.png)

账户由四个部分组成：

- **随机数**
- **账户目前以太币余额**
- **账户的合约代码**
- **账户的存储，存储合约中的状态变量的值**

#### 交易

![](pic\26.png)

![](pic\27.png)

### NFT

**<u>*非同质化代币* ，ERC721, ERC1155</u>**

ERC721:合约起码要实现基本的接口就能算ERC721

例：![](pic\28.png)

## Defi

***去中心化金融，点对点金融服务的总称***

使用场景：

- **去中心化交易所 (DEX)**
- **链上借贷**
- **稳定币**

去中心化只是改变了传统金融服务形式（银行，证券，保险，信托，基金），并不是新的金融需求

![](pic\31.png)

![](pic\29.png)

去中心化金融：建立在区块链，智能合约平台上的金融类产品工具，解决了信任，低门槛，效率，隐私性等问题

例：***跨境转账***

![](pic\30.png)

### DEX 去中心化交易所

DEX(Decentralized Exchange)，即***去中心化交易所，不依赖券商、银行等金融机构提供金融工具，而是利用区块链上的智能合约就可以提供交易服务的交易所。例如Uniswap，Curve，Pancake等等。***

>  **DEX的不存在买方和卖方的差异，因为其本质也不是购兑换对方的资产**，而是直接**<u>从流动性资产池中换取需要的资产</u>**
>
> ***链上的智能合约***，不托管用户资产 + ***<u>透明</u>***

订单薄：挂一个单，等待有人接单或者自己选择附近的价格成交。上面是卖的，下面是买的，中间是实时价格

![](pic\32.png)

> 中心化交易所的卖方在交易所**挂单，买方通过交易所自动撮合下单购买。**
>
>  **DEX的不存在买方和卖方的差异，其本质也不是购兑换对方的资产**，而是直接**从流动性资产池中换取需要的资产。 流动性资金池(liquidity pools)**：用户交易使用的资产池，其中的资金由用户提供，提供资金的行为被称为提供流动性(add liquidity)。为了保证交易稳定、流动性充足，**提供流动性的用户(liquidity provider)可以从每笔交易的手续费里提取分成作为奖励。**

![](pic\38.png)

#### AMM

***使得买方和卖方之间不需要任何信任关系以及第三方就可以安全完成交易***

Auto Market Maker 机制，**自动化做市商**。AMM应用在DEX中，使得买方和卖方之间不需要任何信任关系以及第三方就可以安全完成交易。

> AMM用户的**买/卖本质上是在对相应资产的流动性资金池做兑换**，而非买卖双方的动作。所以AMM天生适合去中心化交易所使用，只要资金池和定价模式完全依托去中心化平台，就没有任何单独实体可以操纵这个系统，同时每个人都可以在这个基础上建立新的应用。

![](pic\33.png)

##### LP token

**LP(流动性提供者)**

LP token即(Liquidity Provider Token)，对于AMM来说，x - y两种资产的兑换需要一个x - y 的资金池，而x，y两种资产由用户提供。用户把x和y资产质押进流动性以获取流动性挖矿(yield farming )奖励，而质押的凭证就是AMM给流动性提供者签发的一种新代币，称为LP token。  

在DeFi飞速发展的过程中，DeFi的术语也在不断。很多情况下LP token具体叫什么是随着项目变化的。比如在Balancer中，通常被称为BPT或者pools tokens；在Uniswap里被称为liquidity tokens或者pool tokens，在Curve中被称为LP token。

![](pic\34.png)

AMM发展至今已经落地在不少应用中，虽然具体实现方法和应用场景略有差别，但很多以简单模型为基础改进，本节将以其代表项目**UniswapV2**为例讲解。各种改进算法将不做讨论 AMM实现功能靠以下两个核心功能：

![](pic\35.png)

![](pic\39.png)

##### swaps

**<u>一个池子内，x越少，就可以用x换取更多的y，反之亦然，套利者会自发将两者价格与全链同步</u>**，这就是swap的基本原理

**Liquidity Pools:**

AMM的用户不直接与另一个用户发生交易，他们的交易都要通过liquidity pools实现，而且因为在AMM的恒定乘积公式算法中，当流动性池内资金量少时，很容易出现x，y有一个非常小导致价格离谱的情况。  **<u>流动性资金池由用户质押资产提供，用户质押资产进入liquidity pools，然后获得质押凭证LP token，而提供流动性的用户也会获得手续费收益，即LP token分红和对应交易所的治理代币</u>**。用户也可以使用LP token赎回其质押的资产，被称为移除流动性(remove liquidity)。

### 去中心化借贷

- **提高杠杆率**
- 币本位主义者

![](pic\36.png)

#### **AAVE**

AAVE是Aave协议的本机管理令牌。基于以太坊的加密货币的持有人可以讨论和投票影响项目方向的提案。

***由于Aave是领先的去中心化金融协议之一，按市值计算，AAVE代币是最大的DeFi代币之一。以太坊投资者可以通过Aave以分散的方式轻松地借入和借出他们的加密货币。***

任何现代金融生态系统的核心都是个人可以借贷其资产的媒介。借贷使人们可以利用其资本来完成任务，而借贷则可以使人们从其本来闲置的资本中获得定期和安全的回报。

加密货币开发商已经意识到了对此类服务的需求，从而启动了所谓的货币市场。Aave是这些市场中最大，最成功的市场之一。

**什么是Aave？**

AAVE是一个 复仇为基础的货币市场，用户可以借入和借出各种数字资产，从 [stablecoins](https://www.wwsww.cn/Stablecoin/6254.html)到altcoins。Aave协议由AAVE持有者管理。

如果不了解底层的Aave协议，将很难理解AAVE令牌是什么，所以让我们深入。

**ETHLend**

Aave的起源可以追溯到2017年.Stani Kulechov和一个开发团队于2017年11月在首次代币发行（[ICO](https://www.wwsww.cn/ico/13596.html)）中发布了ETHLend 。该想法是通过允许用户发布贷款请求并允许用户彼此借贷来借用加密货币。提供。

尽管ETHLend是一个新主意，但该平台及其代币LEND失去了进入2018年熊市的吸引力。该平台的主要痛点是缺乏 流动性以及难以将贷款请求与要约相匹配。

因此，在2018年和2019年的熊市中，ETHLend团队对其产品进行了大修，并于2020年初发布了Aave。

库莱霍夫在播客中说，熊市是ETHLend可能发生的最好的事情之一。这是指他和他的团队有机会修改分散式加密货币贷款的概念，创造了我们现在称为Aave的机会。

**Aave的运作方式**

新改进的Aave在概念上与ETHLend类似。两者都允许以太坊用户获得加密货币贷款或通过借出所持资产获得回报。但是，它们的核心是不同的。

Aave是一种算法货币市场，这意味着贷款是从总库中获得的，而不是单独与贷方匹配。

收取的利率取决于池中资产的“使用率”。如果池中几乎所有资产都被使用，则利率很高，以诱使流动性提供者存入更多资本。如果池中几乎没有资产被使用，则收取的利率很低，无法吸引借款。

Aave还允许用户以不同于他们所存入的加密货币的形式借出贷款。例如，用户可以存入以太坊（ETH），然后提取稳定币存入 [Yearn.finance（YFI）](https://www.wwsww.cn/qtszb/7245.html)以赚取固定收益。

像ETHLend一样，所有贷款都是 超抵押的。这意味着，如果一个人想通过Aave借入价值100美元的加密货币，他们将需要存入更多的钱。

由于加密货币的波动性，Aave包括清算程序。如果您提供的 抵押品低于协议规定的抵押率，则您的抵押品可能会被清算。请注意，清算时要收费。发布抵押品之前，请确保您了解将资金存入Aave的风险。

**其他主要特点**

Aave正在将其范围扩展到不仅仅是货币市场。该平台作为DeFi用户可以获得快速贷款的地方而受到欢迎 。

通常，Aave的货币市场池中的流动性要比借款人所需的贷款多得多。这些未使用的流动性可以被那些接受快速贷款的人使用，这些是无抵押贷款，仅存在于一个以太坊区块的范围内。

基本上，快速贷款允许用户在不发布抵押品的情况下借入大量的加密货币，然后在同一笔交易中归还贷款（只要他们支付一次性利息）。

这允许那些没有大量资金的人 套利并制定其他机会-所有这些都在单个区块链交易中进行。例如，如果您看到以太坊在Uniswap上以500 USDC的价格在另一个去中心化交易所上 以505 USDC的价格交易，则可以尝试借入大量USDC并进行快速交易来套利价格差。

除了提供快速贷款和其他功能外，Aave还致力于开发一款名为[Aavegotchi](https://www.wwsww.cn/qkl/7196.html)的 不可替代令牌（NFT）游戏。

### 稳定币

***由于加密货币以法币计价的价格波动过于剧烈，用户希望能有一个价值稳定的载体***

![](pic\37.png)

## Web3应用

Web3.0结合了**<u>去中心化和代币经济学等概念，基于区块链技术的全新的互联网迭代方向</u>**

随着区块链、数字资产等行业发展

### Defi

- Uniswap
- curve
- dydx
- compound

### NFT

***非同质化代币，一种特殊的加密资产，每一个代币都是独一无二的***

NFT，全称为Non-Fungible Token，指非同质化通证，实质是区块链网络里具有唯一性特点的可信数字权益凭证，是一种可在区块链上记录和处理多维、复杂属性的数据对象。

一般用于数字资产的所有权：

- **艺术品**
- **球星卡**
- **录音**
- **虚拟房地产**
- **宠物**

主要应用：

- **头像类：CryptoPunk**
- **NFT公链：Flow**
- **交易场所：Opensea**

![](pic\40.png)

### DAO

***去中心化自治组织，建立在区块链上，将治理规则以智能合约形式编码***的去中心化自治组织

![](pic\41.png)

> **DAO HACK事件**
>
> 2016年6月17日，DAO因一系列漏洞而被黑客入侵。**黑客利用了一个月前公开的DAO程序代码（不是以太坊协议）中的漏洞。黑客偷走了约360万个以太币（约合5000万美元）**。
>
> 然后关闭DAO。许多投资者扬言要损失全部投资。由于坏消息，以太坊价格从20美元跌至13美元。为了恢复投资者的信心，以太坊社区不得不做出艰难的决定。开发人员要么遵循" 代码就是法律 "标准，就不会扭转360万以太的黑客攻击。另一方面，可以执行硬分叉并使黑客区块链的"旧"部分无效。围绕Vitalik Buterin的以太坊核心开发团队经过长时间讨论后决定进行艰难的决定。因此，被盗的醚可以退还给所有者。
>
> 然而，该决定在以太坊社区中也遭到严厉批评。根据标准"法规就是法律"，一些成员认为硬叉的想法是不道德的。毕竟，发生在区块链上的一切都应该是不变的。
>
> 但是最终实现了硬叉。"旧的"区块链成为以太坊经典。接下来是新的以太坊区块链的约90％。但是，至少有10％的矿工加入了以太坊经典。

### DID身份

***去中心化身份，是一种没有中心化机构做最终担保的数字身份***

**应用层**

![](pic\43.png)

**数据层**

![](pic\44.png)

**重点项目**

- ENS
- SPACE ID
- **Galxe**

![](pic\45.png)

![46](pic\46.png)

## 跨链与多链

![](pic\47.png)

跨链重点项目：

**Wormhole：支持多种跨链转移**

> **跨链原理：使用两个智能合约，接受两个链的token，锁住一个链，然后在另一个链上产生，即模拟了资产转移**
>
> 缺点：效率低，容易被黑客攻击

![](pic\48.png)

## Chainlink

以太坊上第一个**去中心化预言机**解决方案，**<u>*更像区块链与现实世界数据交互的翻译官*</u>**

![](pic\49.png)

**应用场景**

![](pic\50.png)

## 区块链的云服务平台

重点项目：

- **Alchemy**
- **infura**

云服务平台***提供接口，为区块链开发者提供开发服务***

优点：

- **简化去中心化开发**
- **促进对web3访问**
- **提供安全性，稳定性和扩展性**

缺点：

- **增加中心化风险**
- 平台承担流量太大



![](pic\51.png)