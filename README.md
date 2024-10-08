# Trabalho Final de IA
Este projeto tem como objetivo integrar a classificação de imagens, uma aplicação amplamente utilizada em redes neurais, em um contexto lúdico, como o jogo da memória. Para alcançar esse propósito, foram utilizados dois modelos distintos: uma Rede Neural Convolucional (CNN) e um Perceptron Multicamadas (MLP). Essa abordagem permite realizar uma análise comparativa do desempenho de ambos os modelos na tarefa de classificação, destacando suas vantagens e desvantagens dentro do ambiente do jogo.

### Como rodar o jogo
- É importante configurar um ambiente virtual na pasta onde estarão seus arquivos.
    - para Linux: "python3 -m venv nome_do_ambiente" e depois "source nome_do_ambiente/bin/activate"
    - para Windows: "python -m venv nome_do_ambiente" e depois "nome_do_ambiente\Scripts\activate"
      
- Instale as dependências executando "pip install -r requirements.txt" no seu ambiente virtual

- Baixe os modelos treinados da CNN e MLP

- Coloque os arquivos "jogo_MLP.py" e jogo_CNN.py" na mesma pasta dos modelos treinados e execute

### Importante
Quando o jogo é compilado, aparecem alguns warnings e mensagens do próprio sistema, mas isso não indica nenhum erro. Além disso, o MLP demora mais para compilar e gerar a interface do jogo, mas basta ter paciẽncia que no final dá tudo certo!

### Adendo
O modelo treinado do MLP, que consiste em dois arquivos ("mlp_model.pkl" e o "scaler.pkl"), foi muito grande para colocar aqui. Portanto, foi colocado nesse drive:
https://drive.google.com/drive/folders/1NCcnNLeifg1WdX2CCC58eiHY3XpwYzeB?usp=drive_link
