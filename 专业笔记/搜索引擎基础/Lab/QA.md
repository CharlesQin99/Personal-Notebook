Q：返回的结果列表长度？

A：不要求返回全部的文档列表，一般而言如果每页包含10个搜索结果的话，返回3页以上即可。



Q：司法搜索引擎的类案检索功能实现？

A：除了关键词检索，司法搜索引擎需要实现***类案检索功能***，简单而言其实就是长文本检索，可以实现的途径包括①上传司法文档（一般指全文字段，当然要是上传xml也行）的文件进行检索或②提供一个长文本编辑区粘贴类案文本。



Q：知识搜索引擎的知识推理实现？

A：简单一点的话，可以实现一些输入的语句模板，然后指定语句模板输入时就能解析，例如输入“河南的省会是”，即可通过解析返回实体是河南，属性是省会，然后再到数据集中检索；此外也可以考虑直接对查询进行分词，然后在数据集中检索，例如这个句子中只检索到一个实体河南和一个属性省会，那么需要检索的目标也就明晰了。

当然也允许使用一些更复杂的方法，例如句法解析，引入外部数据集进行深度模型训练（如https://github.com/CLUEbenchmark/KgCLUE）等等。此外，也允许使用文心一言、chatgpt等API，但要求“知识”还是来自于我们的数据库，不建议直接接入API进行回答。例如可以用chatgpt设计一些prompt来将“河南的省会是”解析为“实体：河南；属性：省会”，并且协助生成自然语言回答等。

大作业文档中这部分功能给出的分数为15-25分，如果实现的比较简单，例如拿15分的话，可以在其他部分多拿分数。反之为了减轻大家的压力，如果在知识推理实现上做的比较复杂，就可以不做太多其他部分的功能了。



Q：数据集的说明？

A：在网盘中提供了司法数据集和知识检索数据集的简要说明。



Q：推荐的搜索框架？

A：elastic,whoosh, solr, lucene, pyterrier, pylucene, pyserini等，所有开源框架都允许使用，此外如果是BM25算法的话，其实自己手写也很简单。



Q：前端ui的美观程度和得分的关系大吗，还是说做出了多少功能比较重要？

A：功能比较重要，美观的话主要是要求页面清楚使用流畅，用户体验好，会影响主观分。

