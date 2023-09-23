import json
import matplotlib.pyplot as plt

# Nome do arquivo JSON
arquivo_json = 'resultados.json'

# Abre o arquivo JSON e carrega o conteúdo como uma lista de dicionários
with open(arquivo_json, 'r') as arquivo:
    lista_de_dicionarios = json.load(arquivo)

# Métricas que você deseja plotar
metricas = ["Dice", "Fit", "Size", "Position"]
valores = ["Min", "Max", "Media", "Desvio Padrao"]

# Crie uma lista de rótulos com os nomes dos arquivos
nomes_dos_arquivos = [item["Nome do Arquivo"] for item in lista_de_dicionarios]

# Loop sobre os valores
for valor in valores:
    # Crie um novo gráfico para cada valor
    plt.figure(figsize=(10, 6))

    # Loop para adicionar uma linha para cada métrica
    for metrica in metricas:
        valores_metrica = []

        # Coleta os valores da métrica para cada amostra
        for item in lista_de_dicionarios:
            valor_da_metrica = item[valor][metrica]
            valores_metrica.append(valor_da_metrica)

        # Adicione uma linha para a métrica atual
        plt.plot(valores_metrica, marker='o', linestyle='-', label=f'{metrica}')

    # Configurações do gráfico
    plt.title(f'Valores por {valor}')
    plt.xlabel('Amostras')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=f'Valor: {valor}')
    
    # Associe os nomes dos arquivos às amostras no eixo x
    plt.xticks(range(len(nomes_dos_arquivos)), nomes_dos_arquivos, rotation=45)

    # Exibe o gráfico
    plt.tight_layout()
    plt.show()







