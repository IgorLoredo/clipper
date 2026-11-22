# Contexto de trabalho do Clipper

O Clipper seleciona trechos de vídeos longos e renderiza clipes para redes sociais. O trabalho recente concentrou-se na seleção editorial de gameplay, na revisão visual pelo Codex e nos layouts de exportação. A branch `desenvolvimento` reúne essas mudanças no merge `9996ed4`.

## Histórico das branches locais

| Branch | Referência nesta integração | Trabalho incorporado |
| --- | --- | --- |
| `melhoria/identidade-visual-e-skill-do-cursor` | `0faebf0` | Primeira revisão visual de gameplay. Esse commit é a base comum de `main` e `desenvolvimento`. |
| `desenvolvimento` | `fe04af5` e `b6c1ef3` | Identidade ViraCortes, exportação vertical dividida, prévia de renderização, seleção por reação e pontuação editorial de gameplay. |
| `main` | `e97ac27`, `ee787ac` e `6583a59` | Revisão visual independente do provedor de seleção, retomada via `codex-status.json`, templates JSON reutilizáveis e compatibilidade com o layout legado. |

O merge foi feito na `desenvolvimento`. A suíte passou com **123 testes**. Ao registrar este contexto, o merge estava local e `origin/desenvolvimento` ainda apontava para `b6c1ef3`. A pasta `out/` é ignorada pelo Git e guarda os artefatos de execução no checkout.

## Seleção e revisão de gameplay

O Clipper combina transcrição, sinais de reação e medidas do áudio para propor candidatos. O arquivo `gameplay_audio_events.json` guarda os tempos e as medidas de intensidade, variação de pitch, ritmo de fala e pausas. Como o áudio mistura voz, jogo e música, essas medidas servem como pistas para localizar a reação.

O ranking padrão pondera **entretenimento (35%)**, **intensidade (18%)**, **quebra de expectativa (17%)**, **narrativa (15%)**, **clareza fora de contexto (10%)** e **ritmo (5%)**. Os limiares são 65 na nota geral, 70 em entretenimento e 45 tanto em narrativa quanto em clareza fora de contexto. O apoio acústico pode acrescentar até 3 pontos após a aprovação editorial. Use `--gameplay-criteria` para ajustar os pesos, os limiares e o campo `acoustic_bonus`.

A seleção investiga os oito segundos anteriores à reação para localizar sua causa e ajusta o intervalo ao contexto, ao desfecho e à duração solicitada. A revisão visual combina quadros, falas e pistas acústicas com timestamps, avalia os primeiros cinco segundos e pode recuperar candidatos abaixo do limiar editorial. Vídeos sem faixa de áudio podem passar pela seleção visual quando ela está habilitada.

- `--query "descrição da cena"` busca momentos em linguagem natural, inclusive em janelas de fala sem reação marcada.

- `--select-only` grava os cortes em `highlights.json` para uma renderização posterior.
- Quando a execução retorna código 3, `codex-llm/codex-status.json` registra a etapa, o comando e os pedidos pendentes. Cada `*.request.json` recebe um `*.response.json` correspondente antes de repetir o comando.
- Os relatórios `*.candidates.json` e `*.visual.candidates.json` registram as evidências, os intervalos e os vereditos das respectivas etapas. O manifesto `clips.json` associa os arquivos renderizados aos tempos do vídeo original.

## Layouts e abertura

| Opção | Resultado |
| --- | --- |
| Padrão ViraCortes | Saída vertical com a gameplay inteira em uma moldura horizontal, fundo desfocado, título e legendas. A configuração fica em [viracortes.json](../assets/templates/viracortes.json). |
| `--gameplay-split` | Saída 9:16 com a webcam no topo, o jogo recortado ao centro embaixo e as legendas na divisão. `--webcam-crop X,Y,W,H` permite fixar a região da webcam. |
| `--template <arquivo.json>` | Aplica um template reutilizável ao lote, como [social-credit.json](../templates/social-credit.json), com controle de fundo, título, legendas, marca e crédito. |

No gameplay dividido ou renderizado com `--template`, a chamada de abertura usa o gancho editorial, com o título como alternativa, e ocupa até 5,2 segundos. As legendas faladas começam após a abertura. Se o título estiver habilitado no template, também aparece depois da chamada. As legendas usam tamanho 70 no layout dividido e 100 no estilo padrão e no template `social-credit.json`.

A [capa do ViraCortes para Facebook](../assets/references/viracortes-capa-facebook.png) está disponível junto às referências de identidade visual.

Exemplo de prévia de um corte já selecionado, a dois segundos do início:

```powershell
python -m clipper render "out/<trabalho>" --gameplay-split --preview-only --preview-at 2
```

## Lote do Alanzoka

A gameplay [Bem Vindo a Leonida! — GTA 6, Parte 1](https://www.youtube.com/watch?v=kVy8zSfuRc0) tem 2h58. A análise cobriu 13 janelas do vídeo. Dezesseis candidatos passaram pelos critérios editoriais e 15 foram aprovados na revisão visual em duas rodadas. Como o YouTube não forneceu dados de replay, a seleção se baseou na transcrição e nos quadros.

Sete clipes já estavam prontos com a moldura ViraCortes quando o formato mudou. Os oito restantes foram exportados em vertical dividido, usando a mesma região da webcam no vídeo de origem de 1920 × 1080 (`1514,464,398,270`). Os 15 MP4s finais foram conferidos em 1080 × 1920, com áudio e duração compatíveis com o manifesto.

O lote fica localmente em `out/bem-vindo-a-leonida-grand-theft-auto-6-parte-1-kvy8zsfurc0/clips/alanzoka/`. O `clips.json` nessa pasta contém os títulos, as pontuações, os intervalos e o layout de cada clipe.

Para a referência completa dos comandos, consulte o [README do projeto](../README.md) e a [skill clipper-cortes](../.agents/skills/clipper-cortes/SKILL.md).
