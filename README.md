# Vicuna-LangChain-ChatIO
El objetivo de este proyecto es servir un modelo Vicuna que utilice una base de conocimiento local en español, a través de una interfaz web. 

Para ello nos hemos basado en dos proyecto:
- [FastChat](https://github.com/lm-sys/FastChat), que ofrece el servidor web con soporte para Vicuna.
- [Vicuna-LangChain](https://github.com/HaxyMoly/Vicuna-LangChain), que añade una base de conocimiento personalizada, en inglés y chino.

Remplazamos los embeddings y creamos nuestro propio rastreador utilizando la técnica de [Retrieve&Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) con modelos entrenados en español:
- embeddings: [sentence_similarity_spanish_es](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es)
- corss-encoder: [mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)