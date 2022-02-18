# Acabe com o MITO do DS Gênio

Aula 04 - 17 de Fevereiro de 2022 às 20h

# ‼️ Chegou o dia

---

- ~~Entendemos a importância dos dados~~
- ~~Área multidisciplinar (profissão mais sexy do século)~~
- ~~O dia a dia da profissão~~
- ~~Construir um plano de estudos focado no seu objetivo~~
- ~~Entendemos a importância dos projetos em Ciência de Dados~~
- ~~É uma maratona. Ative a disciplina!~~
- ~~Utilizamos bibliotecas para análise de dados e visualização~~
- ~~Como estruturar um projeto em Ciência de Dados~~
- ~~Desmistificar tabus da área (não caia em armadilhas)~~
- ~~Contar história com Dados~~
- Dominar um fluxo de projeto em Ciência de Dados.
- Como se posicionar no mercado de dados em 2022.
- Entender com funciona um dos principais algoritmos de Clusterização (Segmentação)
- Implementar um algoritmo de Machine Learning usando biblioteca Scikit-Learn.
- Subir seu projeto no GitHub
    
    ### 🟢 1 vaga no DataClass
    📚 3 livros
    
    *grupo do whats*
    
    ## Chegou a hora de desbloquear, se prepara! 🔓
    

# 🎯 E ai, valeu a pena participar do Plano Imediato?

---

- ~~Utilizar bibliotecas para análise de dados e visualização~~
- ~~Entendemos a importância dos dados~~
- ~~Área multidisciplinar (profissão mais sexy do século)~~
- ~~O dia a dia da profissão~~
- ~~Construir um plano de estudos focado no seu objetivo~~
- ~~Entendemos a importância dos projetos em Ciência de Dados~~
- ~~Como estrutura um projeto em Ciência de Dados~~
    
    ![03 a 06 de janeiro de 2022 I 20h07.png](Acabe%20com%20%20d3f8b/03_a_06_de_janeiro_de_2022_I_20h07.png)
    

# 📊 Etapa 4 - As máquinas pensam?

---

---

- Quando você ouvir falar sobre de Machine Learning a primeira vez, o que você pensou?
    - Muitas pessoas imaginam um Exterminador do Futuro ou até mesmo algum robô que possa nos ajudar em tarefas do dia a dia.
    - Sim, esse é o caminho que PODEMOS chegar.
        - **Não conseguimos** chegar a um modelo computacional semelhante ao cérebro humano.
    - Tudo isso parece um pouco futurista, mas as aplicações que utilizam **Machine Learning** para tomada de decisão já está entre nós há tempos.
    - A primeira aplicação que realmente se popularizou foi o **Filtro de Spam.**
        - Tá, não era um robô consciente, mas a solução foi classificada como Machine Learning.
        - Mas o que é mesmo **Machine Learning**?
            
            > *Machine Learning é o campo de estudo que possibilita aos computadores a habilidades de aprender sem ser explicitamente programado. 
            *****- Arthur Samuel, 1959
            > 
        - Qual é **normalmente** a abordagem para de desenvolvimento?
            
            ![Captura de Tela 2021-10-19 às 17.11.18.png](Acabe%20com%20%20d3f8b/Captura_de_Tela_2021-10-19_as_17.11.18.png)
            
        - Abordagem usando Machine Learning
            
            ![Captura de Tela 2021-10-19 às 17.11.45.png](Acabe%20com%20%20d3f8b/Captura_de_Tela_2021-10-19_as_17.11.45.png)
            
        - Adaptando-se sem a intervenção (sem ser explicitamente programado).
            
            ![Captura de Tela 2021-10-19 às 17.21.00.png](Acabe%20com%20%20d3f8b/Captura_de_Tela_2021-10-19_as_17.21.00.png)
            
        - Exemplos de aplicações que usam Machine Learning?
            - Análise de produtos de uma linha de produção;
            - Detecção de tumores;
            - Classificação de notícias;
            - **Segmentação de clientes;**
            - Aplicativo com reconhecimento de voz.
- Não reinvente a roda, conheça as ferramentas que você tem disponível.
    - Qual problema você precisa resolver?
        - NOSSO problema → **Clusterização** (Agrupamento)
    - O que é mesmo uma Clusterização?
        - Agrupar um conjunto de objetos de forma que os semelhantes estejam no mesmo grupo.
        - Similiaridade → Relacionamento entre dois objetos.
    - Qual o tipo de Machine Learning
        - **Aprendizado Supervisionado**
        - **Aprendizado Não Supervisionado**
            - Não tenho um “target”.
            - Os dados de entrada descrevem os dados (clustering)
        - **Aprendizado Semi-supervisionado**
    - Mas como isso se aplica ao meu problema?
        - Eu preciso agrupar os clientes do shopping com base em suas informações.
            - Gênero, idade, renda, score
    - Quais são os algoritmos já conhecidos aplicados em problemas de Clusterização?
        - Existem vários algoritmos para clusterização.
        - E no NOSSO problema?
            - K-means
                - Um dos mais populares.
                - Simples e rápido.
                - Se baseia no conceito de Similiaridade → Encontrar semelhantes com base nos atributos.
                - Essa Similiaridade é encontrada através do cálculo de **distância**.
                    - Mas existem vários métodos para cálculo de distância, logo, vai depender da aplicação.
                    - Exclusive Cluster / Overlapping Cluster / Cluster Hierárquico.
                - É um algoritmo do tipo não supervisionado (não possui rótulos).
                - Com o K-means podemos encontrar similiaridade entre os dados e agrupá-los de acordo com o número de clusters (k).
                - A partir de uma iteração, o algoritmo atribui os dados ao grupo que representa a menor distância. -***mais similar***-
                - Precisa entender as etapas e os conceitos! ~~**Veremos isso depois**~~
    - Vamos colocar a mão na massa?
        
        [Google Colaboratory](https://colab.research.google.com/drive/1icK8eyvcoGPXgc2dleYa7dhzAws22gv1?usp=sharing)
        
    - E agora? Empacote!
        
        [pickle - Python object serialization - Python 3.10.1 documentation](https://docs.python.org/pt-br/3/library/pickle.html)
        
    - Começe pelo simples
        - **Streamlit**
            
            [Streamlit * The fastest way to build and share data apps](https://streamlit.io/)
            
    - Tenha + liberdade
        - **Flask**
            
            [Welcome to Flask - Flask Documentation (2.0.x)](https://flask.palletsprojects.com/en/2.0.x/)
            
    - Pense em Caixa, ou melhor, **contêineres**
        
        [Empowering App Development for Developers | Docker](https://www.docker.com/)
        
    - Monte bem sua estrutura, lembre-se, soluções evoluem!
        
        [Dremio | SQL Data Lakehouse Platform for High-Performance BI](https://www.dremio.com/)
        
    
    ### Não menospreze o poder do tempo. Faça um pouco todo dia!
    Fuja do MITO do DS Gênio.
    

# ⏳O que quero que você entenda

---

- Pare de procrastinar sua entrada em uma área tão recompensadora por não saber o que estudar ou como começar. 🎯
- Você não precisa ficar anos estudando pra só então começar.
    - Ative sua disciplina!
- Qual o problema? **Não existe receita mágica!**
    - Você precisa organizar seu aprendizado de forma que já consiga obter resultados.
- As vagas não param de aparecer e **CRIE UM PLANO DE AÇÃO PARA OCUPÁ-LAS**.
    
    ![5.png](Acabe%20com%20%20d3f8b/5.png)
    
- Isso precisa acabar!

# ⚙️ Como funciona a Jornada **#rumoanetuno**

---

- **Conteúdo Gratuito 🎉**
    - YouTube, Instagram, LinkedIn, Roadmap
- **DataClass**: Sua primeira solução em Ciência de Dados
- **Plano Imediato**: Seja Cientista de Dados em 2022📍
- **Método Voyager**: Você vai se tornar Especialista em Dados

# 🚀 Método Voyager

---

- Bônus → DataClass: sua primeira solução em Ciência de Dados
- Desconto Especial → **ÚLTIMA TURMA!**
- Vamos ver um pouco da plataforma?
- Tudo depende do SEU MOMENTO!
- Conheça o Método Voyager
    
    [Método Voyager - Victor Barros - Consultoria em Dados e Inteligência Artificial](https://ovictorbarros.com/metodo-voyager2/)
    

# 📖 Faça sua trajetória e seja visto!

---

- Como construir um portfólio para recrutadores na área de dados?
    1. Busque uma área que você goste ou já possua algum conhecimento (você pode mudar quando quiser)
    2. Entenda as habilidades que você precisa mostrar
    3. Busque conjuntos de dados (Datasets) gratuitos (0800 🤟)
    4. Além do código, escreva sobre todos os insights que você tiver.
        1. Aproveite o notebook!
        2. Abuse da criatividade (tudo é importante)
        3. Comunique os insights obtidos!
    5. Entenda o seu nível e o que precisa mostrar
- **E depois?**
    1. Publique no GitHub (vai rolar uma tarefa massa hein!)
        
        [GitHub: Where the world builds software](https://github.com/)
        
    2. Publique no LinkedIn (mais uma tarefa? 🤩)
        
        [LinkedIn: Log In or Sign Up](https://www.linkedin.com/)
        
    3. Interação com a comunidade (Curta, Comente, Compartilhe)
    
     Esse é o processo, repita! 🔁
    
- Mostra sua capacidade de **resolver problemas** e **aprender coisas novas**
    
    ## O mais importante é o caminho! 🚀
    

# 🎁 Hora do nosso sorteio!

---

- [ ]  Data Science para Negócios →
- [ ]  Como mentir com Estatística →
- [ ]  Inteligência Artificial →
- [ ]  DataClass: sua primeira solução em Ciência de Dados →
