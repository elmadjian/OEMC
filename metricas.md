# Protocolo experimental

### Online Eye Movement Classification Problem



### Datasets

* Nosso (HMR)
* Gazecom
* ~~Santini et al.~~ -> muito pequeno / dados estranhos…



### Treinamento

* extrair do dataset janelas **J** de tamanho **n** contíguas, usando os seguintes dados: (Gaze_x, Gaze_y) e confiabilidade

* extrair a amostra alvo que se deseja classificar relativa à janela. Por exemplo, se **i** for a última posição da janela **J**, então esse alvo poderia ser **i+1** (próxima posição desconhecida) ou **i-5** (*look-ahead* de cinco amostras). Para este experimento, vamos considerar um treinamento para 3 alvos diferentes: próxima posição imediata, -20 ms e -60 ms. Para um dataset criado a 200 Hz, isso seria:
    * i + 1
    * i - 4
    * i - 12
    
    Para um dataset com dados coletados a 250 Hz (GazeCom), isso seria:
    
    * i + 1
    * i - 5
    * i - 15
    
* definida a janela **J**, cada algoritmo/modelo extrai as features que forem consideradas adequadas:
    * direção + velocidade + confiança em multiescala no caso da TCN
    * priors bayesianos no caso do IBD-T?
    
* o treinamento consiste em uma validação cruzada usando 5-fold (i.e., 5 ciclos de treinamento no dataset, em que treinamos com 4/5 e testamos com 1/5 e depois guardamos o desempenho médio de todas as rodadas). Obs.: no caso da TCN o conjunto de validação do treinamento é sempre 1/10 dos dados de treinamento (observando a regra “train/dev/test”). 



### Métricas para avaliação

* F-score em nível amostral para todos os padrões (Fixações, Sacadas, Smooth Pursuits e Blinks)
* F-score em nível de evento com IoU > 0.5 para todos os padrões
* mesmas métricas para dados com subsampling 



-------------------
Da forma como o Rodrigo rodou, estamos descartando 29.5% dos dados do GazeCom.
Para alguns usuários, a situação é mais patológica.
RRP: descarte de 39% 
HHB: descarte de 45%
SSK: descarte de 78% dos dados

Exemplo de distorção da média:
-> score individual pra cada vídeo de RRP:
   FIX F1: 0.9033
   SAC F1: 0.6061
   SP  F1: 0.3383

-> score individual ponderando a quantidade de amostras RRP:
   FIX F1: 0.9055
   SAC F1: 0.5948
   SP  F1: 0.4296

Média total *ponderando* a quantidade de amostras:
Fixation       -> Precision: 0.8015, Recall: 0.9424, F-score: 0.8662
Saccade        -> Precision: 0.9761, Recall: 0.3390, F-score: 0.5032
Smooth Pursuit -> Precision: 0.5316, Recall: 0.3476, F-score: 0.4204

Média total sem ponderar:
    FIX F1: 0.8625
    SAC F1: 0.4953
    SP  F1: 0.4182


