# clipper

Transforma vídeos longos e VODs de lives em cortes verticais (9:16) prontos para TikTok, YouTube Shorts e Reels, com legenda animada palavra por palavra.

```
URL ou arquivo → download (yt-dlp) → transcrição (faster-whisper) → sinais (áudio + chat)
             → seleção de cortes (LLM) → reframe 9:16 (YuNet) → legenda .ass → ffmpeg
```

## Instalação

Requer Python 3.10+. O ffmpeg do sistema é usado se estiver no PATH; senão, o binário do pacote `imageio-ffmpeg` entra no lugar.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # e preencha
```

### Escolha do LLM

Defina o padrão em `LLM_PROVIDER` no `.env` ou troque a cada execução com `--llm provedor[:modelo]`.

| Provedor | Modelo padrão | Chave | Observação |
|---|---|---|---|
| `anthropic` (padrão) | `claude-opus-5-5` | `ANTHROPIC_API_KEY` | saída JSON estruturada |
| `openai` | `gpt-5-mini` | `OPENAI_API_KEY` | JSON schema estrito |
| `gemini` | `gemini-3.6-flash` | `GEMINI_API_KEY` | seleção de cortes em lives longas |
| `deepseek` | `deepseek-flash` | `DEEPSEEK_API_KEY` | JSON Output; usa o SDK `openai` |
| `openrouter` | `google/gemini-2.5-flash` | `OPENROUTER_API_KEY` | uma chave para centenas de modelos |
| `ollama` | `qwen3:8b` | nenhuma | grátis, local: `ollama pull qwen3:8b` |

`pip install -e ".[dev]"` instala os SDKs de todos. Para instalar só o que usar: `pip install -e ".[openai]"` (ou `anthropic`, `gemini`; OpenRouter usa o SDK `openai`; Ollama não precisa de nada).

### Whisper

Sem GPU NVIDIA o padrão é o modelo `small` em CPU (lento em lives longas). Com CUDA o padrão é `large-v3-turbo`. Ajuste com `WHISPER_MODEL`.

## Uso

```bash
clipper run "https://www.youtube.com/watch?v=..." --max-clips 8 --lang pt
clipper run gravacao-da-live.mp4 --chat chat.json
clipper run "https://www.youtube.com/watch?v=..." --content-type gameplay
```

### Podcast, gameplay, anime ou missa

Em `auto` (padrão), títulos com termos de jogo — como `gameplay`, `GTA`, `Fortnite` e `jogando` — usam o layout de gameplay. Títulos de anime e de missa também são reconhecidos. Esses três modos mantêm a tela inteira dentro do vídeo vertical, com fundo desfocado: HUD, arte, legendas já presentes, altar e congregação não são cortados. Os demais vídeos usam o enquadramento atual de podcast, que acompanha o rosto.

Em caso ambíguo, escolha manualmente `--content-type gameplay`, `--content-type anime`, `--content-type mass` ou `--content-type podcast`. O `clips.json` de cada lote registra o tipo aplicado.

Para anime, a seleção favorece diálogo marcante, revelação e virada emocional sem inventar ação visual que não esteja na transcrição. Para missa, privilegia homilia, Evangelho, oração ou reflexão completa, com contexto e linguagem respeitosa. Se o anime já tiver legendas no vídeo, use `--no-captions` para não cobri-las:

```bash
clipper run "URL_DO_ANIME" --content-type anime --no-captions --lang pt
clipper run "URL_DA_MISSA" --content-type mass --lang pt
```

### Mais cortes de 1 a 3 minutos, preservando os anteriores

```bash
clipper run "https://www.youtube.com/watch?v=..." --lang pt --min-dur 60 --max-dur 180 --max-clips 6 --batch longos --exclude-existing
```

O vídeo, a transcrição e os sinais já salvos são reaproveitados. `--exclude-existing` informa à IA os intervalos presentes em `clips/*/clips.json` e descarta sugestões com sobreposição relevante. A duração e a quantidade fazem parte do nome do cache, então cortes curtos não são reutilizados como se fossem longos.

Os novos vídeos vão para `clips/longos/`. Se essa pasta já existir, a próxima será `clips/longos-02/`, depois `longos-03/`, sem sobrescrever os vídeos ou manifestos anteriores. Isso também vale para `render`. Com `--batch`, o `highlights.json` existente é preservado; a nova seleção fica no arquivo `highlights.<llm>-<duracao>-n<quantidade>[...].json`.

### Brief e revisão para campanhas de clipping

Use um brief JSON para informar à seleção o objetivo, público, duração, termos obrigatórios/proibidos, hashtags e regras da campanha. Depois de selecionar os candidatos, uma segunda LLM faz uma auditoria independente. Em `strict` (padrão), só os aprovados são renderizados; em `warn`, todos são renderizados, mas cada `clips.json` inclui a auditoria para revisão humana.

Comece copiando [examples/campaign-brief.example.json](examples/campaign-brief.example.json) e ajuste os requisitos. Exemplo usando uma LLM diferente para revisar:

```bash
clipper run "URL_DO_VIDEO" --campaign campanha.json \
  --llm ollama:qwen3:4b --review-llm gemini:gemini-3.6-flash \
  --review-mode strict --lang pt
```

O arquivo `campaign-review-*.json` no diretório do vídeo registra decisão, nota, motivos, título, legenda e hashtags sugeridos para cada corte. Regras objetivas são conferidas sem LLM: duração, presença literal de `required_terms` na fala e ausência de `forbidden_terms`. A revisão por LLM verifica adequação editorial, contexto, tom e instruções que não podem ser avaliadas apenas por texto literal. Revise o resultado antes de publicar: o revisor não substitui aprovação da marca nem as regras da plataforma.

### Preferência por participante

A seleção pode priorizar as falas de um participante ou aceitar somente falas dele. Essa preferência exige marcações de quem fala em cada intervalo: a transcrição atual do Whisper não identifica vozes nem associa nomes automaticamente.

Use um JSON com tempos em segundos e nomes dos falantes, como [examples/speaker-turns.example.json](examples/speaker-turns.example.json). O exemplo é fictício e deve ser substituído por marcações revisadas do vídeo:

```json
{
  "turns": [
    {"start": 600, "end": 670, "speaker": "Participante A"},
    {"start": 670, "end": 750, "speaker": "Participante B"}
  ]
}
```

```bash
clipper run live.mp4 --prefer-speaker "Participante A" --speaker-turns falantes.json --speaker-mode prefer --speaker-weight 0.25 --min-dur 60 --max-dur 180 --batch foco-participante --exclude-existing
```

- `prefer`: favorece maior proporção de fala do participante, mantendo perguntas e contrapontos necessários. O peso padrão é 0,25: `selection_score = 0,75 × score editorial + 0,25 × percentual de fala preferida`.
- `only`: descarta cortes com palavras atribuídas a outros falantes ou sem identificação. Janelas sem o participante não chamam o LLM. Pode gerar menos cortes ou nenhum.
- `--speaker-weight`: peso de 0 a 1 para ordenar os candidatos. `score` mantém a avaliação editorial; `selection_score` registra o ranking com a preferência.

Trechos não marcados ficam como falante desconhecido. As marcações devem ser não sobrepostas e os nomes precisam coincidir. A preferência não é inferida por opinião política, nomes citados ou frases entre aspas.

#### Gerar marcações locais por voz

Também há diarização local opcional, sem enviar o áudio a uma API. Ela separa vozes em blocos de cinco minutos e compara cada grupo com intervalos curtos de referência que você informa. Use pelo menos uma referência do participante desejado e, quando possível, referências de outros participantes: isso reduz falsos positivos. O modelo só escreve um nome quando a comparação é clara; sobreposições e dúvidas ficam anônimas.

```bash
pip install -e ".[diarization]"
clipper diarize out/<titulo-do-video> \
  --segmentation-model models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
  --embedding-model models/diarization/wespeaker_en_voxceleb_resnet34.onnx \
  --anchor "Participante A=600:620" \
  --anchor "Participante B=900:920" \
  --output out/<titulo-do-video>/falantes.json
```

Os intervalos de referência precisam conter principalmente uma pessoa falando. O JSON resultante registra as referências, similaridades e limitações; revise-o antes de publicar. Depois, passe-o normalmente em `--speaker-turns`.

O cache fica separado por participante, modo, peso e conteúdo das marcações. Os novos cortes registram `speaker_focus`, com a proporção de duração das palavras atribuídas ao participante, o tempo de fala de outros/desconhecidos e a origem das marcações. Os arquivos anteriores são preservados.

### DeepSeek e troca automática de modelos

Preencha no `.env`:

```dotenv
DEEPSEEK_API_KEY=sua_chave
DEEPSEEK_MODEL=deepseek-flash
LLM_PROVIDER=deepseek:deepseek-flash,deepseek:deepseek-v4-pro,gemini:gemini-3.6-flash
```

Ou escolha a cadeia apenas para uma execução:

```bash
clipper run live.mp4 --llm deepseek:deepseek-flash,deepseek:deepseek-v4-pro,gemini:gemini-3.6-flash
```

O suporte direto ao DeepSeek usa [JSON Output](https://api-docs.deepseek.com/guides/json_mode/) e pode ser instalado com `pip install -e ".[deepseek]"`. Chave ausente, saldo esgotado, limite de uso, modelo indisponível e saída truncada por limite de tokens fazem a cadeia tentar a próxima entrada. O progresso de cada janela concluída fica salvo. Modelos do mesmo provedor podem compartilhar saldo: nesse caso, é preciso ter outro provedor configurado com saldo ou cota disponível.

### Selecionar cortes no Codex

Abra este repositório no Codex e peça à skill `$clipper-cortes` para gerar
cortes de um arquivo local ou de uma URL. A [skill do projeto](.agents/skills/clipper-cortes/SKILL.md)
descreve como o Codex responde aos pedidos e retoma a execução. Para analisar um
vídeo do YouTube desde o início, use a URL sem o parâmetro `t`.

Para iniciar pelo terminal, ative o ambiente Python do projeto e execute:

```bash
python -m clipper run live.mp4 --llm codex
```

O código de saída `3` indica que o Clipper aguarda os pedidos JSON listados em
`out/<video>/codex-llm/` antes de renderizar.
Para cada `*.request.json` listado, a skill lê o prompt e grava o objeto definido
pelo schema em `*.response.json` com o mesmo identificador. Ao repetir o comando,
o Clipper valida as respostas e aproveita as janelas já concluídas. Com
`--campaign`, o mesmo fluxo também atende à revisão antes da renderização.
O modo Codex dispensa chave de API de LLM no Clipper.

Para vídeos de gameplay, há um atalho:

```bash
python -m clipper gameplay "video.mp4" --llm ollama:qwen3:4b --lang pt --max-clips 6
python -m clipper gameplay "video.mp4" --llm gemini:gemini-3.6-flash --visual-review-llm codex
```

O provedor escolhido em `--llm` faz a seleção inicial. Depois, o revisor visual
independente envia ao Codex uma grade de seis quadros por trecho, incluindo o
pico local mais forte. A shortlist tem até 12 candidatos da transcrição e pode
receber trechos adicionais sugeridos por picos de áudio, movimento ou mudança
de cena. A revisão procura um gancho visual, ação compreensível e um desfecho,
inclusive em trechos silenciosos.
Os quadros são extraídos localmente. Abrir as grades no Codex acrescenta
contexto visual à conversa. Use `--no-visual-review` para pular essa etapa. A
revisão visual não obriga a seleção a usar Codex: por exemplo, Ollama pode criar
a shortlist localmente e Codex pode revisar somente os finalistas.

Quando o processo aguarda o agente, `codex-status.json` registra a etapa atual,
o comando exato, os pedidos pendentes e a próxima ação. Ao terminar, o estado
muda para `complete`. Isso evita depender apenas do código de saída 3 para retomar.

No Windows, se o Whisper detectar uma GPU mas falhar ao carregar
`cublas64_12.dll`, rode em CPU. Por exemplo, no PowerShell:

```powershell
$env:WHISPER_DEVICE = 'cpu'
$env:WHISPER_MODEL = 'base'
python -m clipper run live.mp4 --llm codex --lang pt
```

Confira os MP4 e o manifesto `clips.json` do lote antes de publicar. As
legendas usam a transcrição automática. Se algum corte sair vazio, compare os
tempos do manifesto com a duração do arquivo de origem.

### Comparar LLMs no mesmo vídeo

A transcrição fica em cache, então só a escolha dos cortes roda de novo:

```bash
clipper run live.mp4 --llm anthropic
clipper run live.mp4 --llm openai:gpt-5
clipper run live.mp4 --llm gemini:gemini-3.6-flash
clipper run live.mp4 --llm ollama:llama3.1:8b
clipper run live.mp4 --llm openrouter:deepseek/deepseek-chat
```

Cada seleção gera um cache com o provedor, modelo, duração e quantidade de cortes no nome. Cada renderização usa uma nova pasta, lado a lado com as anteriores.

### Ollama local

O Ollama pode fazer a seleção inteiramente no computador, sem chave e sem consumo de tokens de APIs. Deixe o servidor rodando e baixe um modelo uma vez:

```bash
ollama serve                 # ou .\\tools\\ollama\\ollama.exe serve, na instalação portátil do projeto
ollama pull qwen3:4b         # ou .\\tools\\ollama\\ollama.exe pull qwen3:4b
clipper run live.mp4 --llm ollama:qwen3:4b
```

O conector usa JSON estruturado, desativa o modo de raciocínio exibido e limita cada resposta a 1.200 tokens por padrão. Em CPU, modelos menores como `qwen3:4b` são a opção mais prática; eles tendem a levar mais tempo que uma API. Combine-o com provedores pagos somente se quiser fallback: `--llm ollama:qwen3:4b,gemini:gemini-3.6-flash`.

### Perfil para notebook e uso econômico de tokens

Por padrão, cada resposta de LLM agora é limitada a 1.200 tokens e pede poucos candidatos por janela; o projeto não gasta milhares de tokens explicando cortes que serão descartados. Ajuste apenas se o modelo estiver truncando JSON:

```bash
clipper run live.mp4 --llm-output-tokens 1600
```

Para rodar inteiramente no notebook, instale o Ollama, baixe uma vez o modelo e use o perfil local. Ele usa `qwen3:4b`, contexto de 8k, saída de 768 tokens e janelas de 5 minutos — configuração mais realista para 16 GB de RAM, sem consumo de API:

```bash
ollama pull qwen3:4b
clipper run live.mp4 --notebook --lang pt --max-clips 6
```

Em notebooks com 8 GB de RAM, prefira `qwen3:1.7b` e transcrição `base`: `WHISPER_MODEL=base` e `clipper run live.mp4 --notebook --llm ollama:qwen3:1.7b`. Os resultados locais tendem a ser menos seletivos que modelos de API; uma combinação econômica é usar o Ollama para rascunhos e Gemini/DeepSeek somente para a seleção final.

### Fallback entre LLMs (cota ou créditos acabando)

Passe uma lista separada por vírgula; se um LLM falhar no meio do vídeo, o próximo assume a partir da janela em que parou:

```bash
clipper run live.mp4 --llm anthropic,openai:gpt-5-mini,gemini,ollama:qwen3:8b
```

- Qualquer erro do provedor (cota/créditos esgotados, rate limit depois dos retries do SDK, chave inválida, SDK não instalado, Ollama fora do ar) tira aquele LLM da cadeia até o fim da execução. Recusa de conteúdo não troca de LLM: só pula a janela.
- Cada janela de 15 min concluída é gravada na hora em `highlights.<llm>.progress.json`, com o LLM que a fez. Nada já respondido é pedido de novo.
- Se todos falharem, o comando sai com código 2 e diz em que janela parou. Rode o mesmo comando de novo (pode trocar a lista depois do primeiro LLM) e ele continua de onde parou.
- O progresso pertence ao primeiro LLM da lista, então `--llm openai` sozinho continua sendo uma comparação limpa. Mudar `--min-dur`, `--max-dur` ou `--max-clips` usa outro cache, preservando o progresso anterior.
- Cada corte em `clips.json` traz o campo `llm` com quem o escolheu.

A lista também pode ficar no `.env`: `LLM_PROVIDER=anthropic,openai,ollama`. Um `--llm` sem vírgula usa só aquele LLM.

Saída em `out/<titulo-do-video>/`:

| Arquivo | Conteúdo |
|---|---|
| `transcript.json` | palavras com timestamps (cache: não transcreve de novo) |
| `signals.json` | picos de áudio e de chat |
| `highlights.<llm>.json` | cortes escolhidos por cada LLM (cache) |
| `highlights.<llm>.progress.json` | respostas por janela e quem respondeu (retomada após falha) |
| `highlights.json` | cópia da última execução (editável) |
| `clips/<llm>/*.mp4` | vídeos 1080x1920, nome `<rank>_<score>_<titulo>.mp4` |
| `clips/<llm>/clips.json` | título, gancho, motivo e score de cada corte, para postar |

### Organização de `out/`

Cada vídeo permanece isolado em `out/<titulo-do-video>/`. Dentro dele, `clips/` contém somente lotes que renderizaram vídeos; tentativas sem resultado ficam em `attempts/`, sem apagar seus registros. Auditorias de preservação ficam em `audit/` e os arquivos de diarização em `diarization/`.

Na raiz de `out/`, use `_logs/<video>/` para logs de execuções concluídas e `_validation/` para previews e verificações. Não mova um log que ainda está sendo escrito: mova-o para `_logs/` somente após o processo terminar. Assim, novos vídeos não se misturam com cortes, logs ou validações de outros vídeos.

### Ajustar os cortes à mão

Edite `highlights.json` (início, fim, título) e renderize de novo em `clips/manual/` sem chamar o LLM:

```bash
clipper render out/<titulo-do-video> --batch revisao
```

### Chat como sinal de momento viral

`--chat` aceita:
- `live_chat.json` do YouTube (o `run` já baixa automaticamente quando a URL é de uma live gravada);
- JSON do [TwitchDownloader](https://github.com/lay295/TwitchDownloader) (`comments[].content_offset_seconds`);
- uma lista JSON de segundos (`[12.5, 13, ...]` ou `[{"t": 12.5}, ...]`).

### Métricas de atenção por trecho

Quando disponível, o `run` coleta o mapa público de trechos mais reassistidos do YouTube, cruza com os sinais de áudio/chat e fornece os picos de replay à IA. O índice é relativo entre 0 e 1: não representa quantidade de visualizações, espectadores únicos ou retenção. As faixas mantêm a resolução fornecida pelo YouTube; não são medições por segundo.

Para analisar um vídeo que já está salvo, sem chamar o LLM ou renderizar novamente:

```bash
clipper metrics out/<titulo-do-video> --source "https://www.youtube.com/watch?v=..."
```

Depois da primeira coleta, `clipper metrics out/<titulo-do-video>` reutiliza os dados locais. O relatório inclui os trechos com maior sinal, quais deles se sobrepõem a cortes existentes e as métricas dos cortes já gerados. Os vídeos e manifestos anteriores são preservados.

| Arquivo | Conteúdo |
|---|---|
| `engagement.json` | faixas do replay público, origem e data da coleta |
| `attention.json` | métricas por faixa, regiões ainda não utilizadas e comparação dos cortes existentes |
| `attention.md` | relatório legível com intervalos e métricas |
| `clips/<lote>/clips.json` | novos cortes incluem `engagement`: replay médio/máximo, cobertura, mensagens no chat e picos de áudio |

Novos dados de replay usam outro cache de seleção. Quando o mapa público não está disponível, a seleção continua com áudio, chat e conteúdo, sem inventar dados de audiência. Esses sinais orientam a avaliação editorial; a melhoria dos cortes ainda deve ser avaliada após a publicação. Retenção detalhada pela [YouTube Analytics API](https://developers.google.com/youtube/analytics/channel_reports#retrieving-a-report) exige OAuth do dono do canal e não está conectada nesta versão.

## Como funciona

- **Seleção**: a transcrição é cortada em janelas de 15 min com 60 s de sobreposição. Cada janela vai ao LLM com os critérios de viralidade (gancho, pico emocional, opinião forte, revelação, conflito, frase de efeito, clímax, valor prático) e com os picos de áudio/chat como dica. As bordas são alinhadas às palavras e estendidas até o fim da frase; cortes sobrepostos são descartados pelo score.
- **Reframe**: amostra 2 frames/s, detecta o maior rosto (YuNet, baixado em `~/.cache/clipper` no primeiro uso) e move a "câmera" só quando o rosto sai de uma zona morta, com transição suave. Sem rosto em pelo menos 40% das amostras, usa o layout com fundo desfocado.
- **Legenda**: grupos de até 3 palavras em caixa alta, palavra atual em amarelo, com "pop" na entrada.

## Testes

```bash
pytest
```

## Uso responsável

Corte apenas conteúdo seu ou com autorização do criador. Repostar lives de terceiros sem permissão gera strike de direitos autorais.

## Créditos de ideias

Abordagem inspirada em [AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator) (MIT) e [OpenShorts](https://github.com/mutonby/openshorts) (MIT). Nenhum código foi copiado.

