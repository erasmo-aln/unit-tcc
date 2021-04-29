# Classificação de placas de trânsito com o uso de CNNs
## Visão Geral do Projeto
Este é o meu TCC do curso de Engenharia Elétrica, da Universidade Tiradentes. Como o título sugere, foi feita a classificação de 43 placas de trânsito usando CNNs (*Convolutional Neural Networks*). Após testar diversas configurações variando alguns parâmetros, o modelo final foi capaz de alcançar 99,66% de acurácia no conjunto de teste, superando o desempenho humano e outros modelos utilizados na competição GTSRB.

*Link para o TCC: [Clique aqui](TCC/TCC.pdf)*

## Índice
- [Informações Gerais](#info)
  - [Origem dos Dados](#origem)
  - [Pacotes Necessários](#pacotes)
  - [Jupyter Notebook](#jupyter)
  - [Trabalhos Futuros](#trabalhos)

<a name="info"></a>
## Informações Gerais
Abaixo será listada uma série de informações sobre o projeto e o repositório, como links para acessar o Jupyter Notebook, os pacotes que foram usados, além de ideias sobre trabalhos a serem realizados.

<a name="origem"></a>
### Origem dos Dados
Basicamente, há duas formas de fazer o download dos dados: diretamente do site do *[GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)*, ou via *[Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)*. Ambos terão os datasets de treino e teste separados. 

Por conta do tamanho dos arquivos, não é possível disponibilizá-los aqui no GitHub. Porém, o arquivo [`create_dataset.py`](create_dataset.py) mostra como transformar todas as imagens (treino e teste separadamente) em um único arquivo `.h5`, que será utilizado para fazer o pré-processamento.

<a name="pacotes"></a>
### Pacotes Necessários
Todas as bibliotecas e suas respectivas versões podem ser vistas no arquivo [`requirements.txt`](requirements.txt). O pacote mais relevante, entretanto, é o *Tensorflow*, que recentemente foi atualizado para a versão 2.3, porém neste artigo foi utilizada a versão 2.1. Alguns métodos entre essas versões são incompatíveis, portanto é necessário atenção.

<a name="jupyter"></a>
### Jupyter Notebook
Além do TCC oficial em formato *pdf*, o arquivo [`TCC.ipynb`](TCC/TCC.ipynb) detalha cada passo de todo o processo, desde o carregamento do *dataset*, construção da CNN, até algumas análises após a construção. 

Caso queira utilizar o mesmo modelo apresentado no meu TCC, ele está no disponível no meu [`Google Drive`](https://drive.google.com/drive/folders/1WhWLESl9XUqLxlqsgOVbyS5_nwOi_GsO?usp=sharing), juntamente com o *notebook* e o *pdf*. 

<a name="trabalhos"></a>
### Trabalhos Futuros
- [ ] Traduzir o `README.md` para inglês;
- [ ] Traduzir o `Jupyter Notebook` para inglês;
- [ ] Atualizar o código para o *Tensorflow 2.3*;
- [ ] Migrar de Jupyter Notebook para scripts em Python.
