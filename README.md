# guardrails-study

Estudo técnico sobre como validar saída estruturada de modelos de linguagem. O recorte deste repositório é instrumental: quatro estratégias de guarda, um cliente de LLM agnóstico de provedor e um runner com orçamento de tentativas. As tarefas são sintéticas e públicas. O repositório não contém resultado experimental.

## Objetivo

Comparar quatro formas de decidir se a saída de um LLM pode ser aceita como JSON de uma tarefa de extração ou formatação:

1. validação somente por JSON Schema;
2. validação por modelo Pydantic, com nova pergunta quando a saída falha;
3. regras determinísticas (expressão regular e invariantes de negócio);
4. um segundo LLM no papel de revisor.

O que se mede, fora da guarda, é a taxa de saída inválida que a guarda aceita, a taxa de falso rejeição contra o gabarito, o número de tentativas, a latência e o custo extra de tokens quando o provedor informa uso. A definição operacional está na seção Avaliação offline. PR1 constrói o registro de cada execução. Este arquivo não mede um modelo real.

`valid` no registro significa "a guarda aceitou a saída final". Não significa "bate com o gabarito". Concordância com o gabarito é métrica do PR3, calculada fora da guarda.

## Hipótese

Falhas de saída estruturada não são um único fenômeno.

- JSON Schema e Pydantic devem barrar JSON quebrado, tipo errado e valor fora de enum ou padrão. Não devem, sozinhos, perceber que um JSON bem formado extraiu o valor errado do texto.
- O modelo Pydantic deste estudo acompanha o schema nas restrições estruturais, recusa coerção (`"1990"` não vira inteiro, `"false"` não vira booleano) e ainda interpreta data como calendário real. A diferença operacional planejada da estratégia 2 é o re-ask: a guarda devolve o motivo e o runner pergunta de novo, até um orçamento.
- Regras determinísticas devem pegar invariantes que o schema do estudo deixa de fora de propósito: data impossível além do formato `YYYY-MM-DD`, produto de campos e teto condicional de SLA. A hipótese é que o falso rejeição delas só ocorre quando a regra está errada ou estreita demais. Isso ainda não foi medido.
- Um segundo LLM pode rejeitar extração infiel ao texto-fonte mesmo quando o JSON passa nas checagens estruturais. A hipótese é que isso reduz a taxa de inválidos que escapam e aumenta tokens, latência e falso rejeição. A direção e o tamanho desses efeitos são perguntas empíricas, não conclusão.

O re-ask só ajuda se o motivo devolvido for acionável e se ainda houver tentativas. Esgotar o orçamento conta como saída não aceita pela guarda.

## Metodologia planejada

Unidade experimental: uma tarefa sintética com instrução, texto-fonte, JSON Schema, modelo Pydantic, lista de regras e gabarito rotulado. O conjunto versionado está em `data/eval/tasks.v1.jsonl` e é descrito na seção Dataset de avaliação. A fixture de 10 tarefas em `guardrails_study/fixtures/tasks.json` continua só para os testes de guarda e do runner. Nenhuma tarefa usa dado pessoal real. E-mails ficam em `example.com`. Identificadores são fictícios (`INV-1042`, `ORD-77`, `TRK-9`; no dataset, `INV-2001`, `ORD-11`, `TRK-301`).

Cada estratégia roda sozinha. Não há empilhamento de guardas: empilhar confundiria a origem da rejeição. O runner anota, por tentativa, a saída bruta, se foi aceita, qual guarda rejeitou, a latência medida com `time.perf_counter` em torno da chamada do gerador e da guarda, e tokens somente quando todas as chamadas da tentativa reportaram uso. O custo em USD só é preenchido se, além do uso, `PRICE_PROMPT_PER_MILLION` e `PRICE_COMPLETION_PER_MILLION` estiverem definidos. Ausência de uso ou de preço fica `null`. O código não inventa contagem de tokens nem preço.

Protocolo de tentativas neste PR, sujeito a revisão no PR4:

- `json_schema`: 1 tentativa (validação somente);
- `pydantic`, `rules` e `second_llm`: até `MAX_ATTEMPTS` (padrão 3). O motivo da rejeição entra na conversa. O gabarito não entra.

As guardas não reparam a saída. Cerca de markdown, vírgula sobrando e aspas simples continuam inválidas para as guardas determinísticas. O revisor LLM não recebe o gabarito; recebe instrução, fonte, schema e candidato. Se a resposta do revisor não for um objeto JSON com `accept` booleano e `reason` string, a guarda rejeita (falha fechada) com o código `checker_malformed`.

Divisão de trabalho entre schema, Pydantic e regras, para as estratégias não serem cópias:

- o schema cobre tipo, `required`, `additionalProperties: false`, enum, padrão, mínimo e tamanho de lista ou string;
- `format` do Draft 2020-12 não é assertado;
- Pydantic espelha essas restrições e trata `due_date` como data de calendário;
- regras repetem checagens locais para o braço funcionar sozinho e acrescentam `calendar_date` (além do formato), `product_equals` e `max_when_eq`. A regra não rejeita campo extra. Schema e Pydantic rejeitam.

Cliente: `complete(messages)` devolve texto e, se houver, uso de tokens. `ScriptedLLM` entrega uma fila determinística. Os rótulos `valid`, `malformed` e `subtle` saem da fixture: `valid` é o gabarito serializado, `malformed` não é JSON, `subtle` é JSON factualmente errado que ainda satisfaz schema, modelo e regras. A fila não se recicla; acabou o roteiro, o fake levanta erro em vez de inventar texto. Os SDKs `openai` e `anthropic` só são importados na primeira chamada ao vivo e ficam num extra opcional. A suíte não usa rede nem chave.

A comparação com o gabarito não entra na guarda. O harness da seção Avaliação offline lê o `RunResult` e calcula as taxas. `examples.subtle`, na fixture de 10 tarefas, existe para o fake emitir o caso em que a decisão da guarda e o gabarito divergem. Não é uma taxa.

## Dataset de avaliação

O conjunto experimental v1 está em `data/eval/tasks.v1.jsonl`: 45 linhas, uma tarefa por linha. O loader é `load_eval_tasks`. Esta seção não reporta resultado de modelo.

Cada linha tem `id`, `category`, `instruction`, `source_text`, `json_schema`, `ground_truth`, `model_key`, `rules` e `examples` com `valid`, `malformed` e `subtle`. O `model_key` é um dos dez modelos Pydantic do PR1. `examples.valid` é o gabarito em JSON canônico (`json.dumps` com `sort_keys` e `ensure_ascii=False`), o mesmo texto que `script_from_labels` emite para o rótulo `valid`. `malformed` não é JSON. `subtle` é JSON que passa em schema, Pydantic e regras e difere do gabarito.

Contagem por categoria primária. A categoria é o estresse que o loader reproduz com uma sonda.

| Categoria | Itens | Sonda |
| --- | ---: | --- |
| `structural_breakage` | 12 | `malformed` não é JSON. O schema tem padrão, enum ou tamanho. Campo extra falha no schema e no Pydantic e passa nas regras. |
| `type_coercion_bait` | 10 | Há um inteiro, número ou booleano. A sonda cita esse valor entre aspas. Schema, Pydantic e regras rejeitam. |
| `calendar_business_rule` | 10 | Há `calendar_date`, `product_equals` ou `max_when_eq`. A sonda passa no schema, falha na regra e, no caso da data impossível, também no Pydantic. |
| `unfaithful_extraction` | 13 | O `subtle` troca um texto que também está na fonte. As três guardas determinísticas aceitam. |

Contagem por `model_key`. Cada tipo do PR1 aparece 4 ou 5 vezes.

| `model_key` | Itens |
| --- | ---: |
| `invoice_total` | 5 |
| `due_date` | 5 |
| `contact_email` | 4 |
| `ticket_priority` | 5 |
| `line_item` | 5 |
| `phone_e164` | 4 |
| `shipment_delayed` | 5 |
| `currency_amount` | 4 |
| `action_items` | 4 |
| `order_status` | 4 |

Coerção e calendário/regra ficam nos tipos com inteiro, número, booleano, data, produto ou teto de SLA. E-mail, telefone, status e lista ficam em quebra estrutural e extração infiel. A contagem 12 / 10 / 10 / 13 segue essa concentração.

Como foi construído: as tarefas foram escritas contra os schemas e as regras já suportados pelo PR1. O texto-fonte e o gabarito foram definidos juntos. Nenhum modelo preencheu o rótulo e não houve segundo anotador. O loader recusa o arquivo inteiro se um id repete, se o gabarito não passa no schema, no Pydantic e nas regras, se o `subtle` não passa nessas três guardas ou não difere do gabarito, se o `malformed` é JSON, ou se a sonda da categoria não se comporta como a tabela. Datas de calendário estão no texto como `DD/MM/AAAA` e no gabarito como ISO; a tarefa infiel de data repete as duas datas ISO na fonte, para o distrator ser literal. Enums de prioridade e status estão em português no texto e em código no JSON; o mapa fica na instrução (`alta=high`, `pago=paid`). Nas tarefas infiéis de status, o código em inglês também aparece na fonte, para o distrator ser literal.

Limitações. São 45 itens sintéticos, de um autor. Isso não estima taxa de erro de modelo. O `subtle` não é a violação de calendário, de produto nem de SLA: por construção ele passa nas guardas determinísticas. A data `2024-02-31`, o total de linha incoerente e o SLA de prioridade `high` acima do teto são sondas do loader, não linhas do roteiro. A isca `"1990"` também não cabe em `malformed`, porque é JSON válido, nem em `subtle`, porque as guardas a rejeitam. Itens de `line_item` carregam `product_equals` mesmo quando a categoria primária é outra. A fixture de 10 tarefas permanece para os testes de guarda.

```python
from guardrails_study.dataset import load_eval_tasks

tasks = load_eval_tasks()
```

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

A configuração é por variável de ambiente. O arquivo versionado é só `.env.example`. Não há `.env` no repositório e a suíte não carrega esse arquivo sozinha. Para um shell local:

```bash
cp .env.example .env
set -a && source .env && set +a
```

`LLM_PROVIDER=fake` não exige chave. Provedores `openai` e `anthropic` exigem modelo e chave. O pacote correspondente entra com `uv sync --extra live`. Os testes não exercitam a rede. Os adaptadores ao vivo usam `temperature=0` e não são cobertos por chamada real.

O trecho abaixo só exercita o fake. Não é resultado de modelo. O teste `test_pydantic_preset_reasks_until_valid` trava o contrato.

```python
from guardrails_study.config import Settings
from guardrails_study.llm import ScriptedLLM
from guardrails_study.runner import run_strategy
from guardrails_study.tasks import load_tasks, script_from_labels

task = next(item for item in load_tasks() if item.id == "invoice_total")
generator = ScriptedLLM(script_from_labels(task, ["malformed", "valid"]))
result = run_strategy("pydantic", task, generator, Settings())
print(result.valid, result.attempts, result.rejected_by)
```

## Avaliação offline

O comando `evaluate-guards` (o mesmo que `python -m guardrails_study.evaluate`) percorre `data/eval/tasks.v1.jsonl`, executa uma estratégia ou `--strategy all`, e grava dois arquivos no diretório de `--output`:

- `records.jsonl`: uma linha por tarefa e estratégia;
- `summary.json`: agregado, quebra `by_strategy` (e, dentro dela, por categoria) e quebra `by_category` somando as estratégias pedidas.

`valid` no registro continua sendo o aceite da guarda. A comparação com o gabarito é feita depois, neste harness, e não volta para o prompt.

```bash
uv run evaluate-guards --strategy all --scenario always_valid --output runs/pipeline-check
uv run python -m guardrails_study.evaluate --strategy json_schema --scenario subtle --output runs/subtle-check
```

`runs/` está no `.gitignore`. Os arquivos `records.jsonl` e `summary.json` são substituídos se o diretório já existir. Não versionar essa saída.

O backend padrão é o fake (`LLM_PROVIDER` vazio ou `fake`). `--scenario` escolhe a fila do gerador. O tamanho da fila é o orçamento da estratégia, e a fila não recicla.

| Cenário | Gerador |
| --- | --- |
| `always_valid` | o gabarito em toda tentativa |
| `malformed_then_valid` | o texto que não é JSON e, na tentativa seguinte, o gabarito |
| `subtle` | o exemplo que passa nas guardas determinísticas e difere do gabarito |

`json_schema` segue com uma tentativa. As outras estratégias usam `MAX_ATTEMPTS`. Em `second_llm` o fake também precisa de um revisor. O revisor scriptado aceita só o JSON igual ao gabarito e rejeita o resto, com motivo genérico (`not faithful to the source`), sem devolver o gabarito no texto de reparo. A guarda real não recebe o gabarito. O duplo não é um modelo: existe para o encanamento ter um veredito determinístico. Por isso, no cenário `subtle`, as três guardas determinísticas aceitam e o `second_llm` scriptado recusa até esgotar o orçamento. Isso é checagem de encanamento. Esses números não são achado sobre modelo e não devem ser citados como desempenho.

O campo `note` do `summary.json`, no fake, repete o aviso em inglês: a saída é `pipeline check`, não achado de modelo.

### Como ler cada métrica

O denominador de toda taxa é `scored_count`: linhas com `status` igual a `ok`. Erro de transporte ou do provedor fica com `status` `error`. Nessa linha, `valid`, `matches_ground_truth`, `escaped_invalid`, `false_rejection`, `budget_exhausted`, tentativas realizadas, latência, tokens e custo ficam null. `max_attempts` ainda guarda o orçamento da estratégia, porque é configuração, não desfecho. A linha entra só em `error_count` e não é saída errada. Se a exceção ocorre no meio das tentativas, a tarefa inteira é erro: não há rejeição parcial. Se `scored_count` é zero, as taxas e as distribuições são null.

- `acceptance_rate`: fração em que a guarda aceitou (`valid` true).
- `escaped_invalid_rate`: fração em que a guarda aceitou e o JSON parseado difere do gabarito. Texto que não parseia conta como diferente.
- `false_rejection_rate`: fração em que a guarda rejeitou e o JSON parseado é o gabarito.
- `budget_exhaustion_rate`: fração em que a guarda rejeitou e as tentativas atingiram o orçamento. Aceitar na última tentativa não esgota.

As quatro taxas não fecham em 1. Recusar JSON quebrado não é falso rejeição nem inválido que escapou.

A igualdade com o gabarito ignora ordem de chave e espaço. Bool não equivale a 0 nem a 1. Int e float com o mesmo valor equivalem (`20` e `20.0`), porque o JSON do dataset não separa esses dois tokens.

`attempts` e `latency_ms` trazem `mean`, `p50` e `p90` sobre as linhas ok. O percentil é nearest-rank: ordena a lista e toma a posição `ceil(p/100 * n)`, com a primeira posição em 1. Numa lista ordenada `[1, 2, 3, 4]`, p50 cai na posição 2 (valor 2) e p90 na posição 4 (valor 4). Com um único valor, p50 e p90 são esse valor.

`prompt_tokens`, `completion_tokens` e `cost_usd` são somas. Cada campo é independente: se alguma linha ok tiver aquele campo null, a soma dele é null. Ausência não vira zero. O fake não informa uso, então essas somas saem null. O preço só aparece se o runner já tiver preenchido `cost_usd` (uso completo e as duas variáveis de preço). O harness não inventa token nem preço.

`by_category` no topo soma as estratégias selecionadas. A categoria dentro de `by_strategy` não soma com as outras.

### Provedor ao vivo

`openai` e `anthropic` só seguem com `--allow-live` e com modelo e chave no ambiente
(`LLM_MODEL` e `LLM_API_KEY`; no revisor, `CHECKER_MODEL` e `CHECKER_API_KEY`). Sem o flag
ou sem a chave o processo termina com código 2 e não envia pedido. A mensagem de recusa
não inclui o valor da chave. O flag não é exigido para o fake. `--scenario` vale só para
o fake. O revisor ao vivo só entra se a estratégia pedida inclui `second_llm` e o gerador
também é ao vivo. Gerador fake com `CHECKER_PROVIDER` openai ou anthropic sai com código 2
mesmo com `--allow-live`: o fake não chama revisor ao vivo. Gerador ao vivo com revisor
fake também sai com código 2, sem pedido. A suíte não percorre chamada real e este README
não traz tabela de modelo.

## Roadmap

- PR2: dataset versionado em `data/eval/tasks.v1.jsonl` (45 tarefas sintéticas com gabarito). A fixture de 10 tarefas permanece para os testes de guarda. Sem métrica.
- PR3: `evaluate-guards` grava `records.jsonl` e `summary.json` com as taxas contra o gabarito. O backend padrão é o fake; o número que ele produz é checagem de encanamento, não resultado de modelo.
- PR4: experimentos com modelos reais e relatório. Sem números antes da execução registrada.

## O que fica de fora

A suíte não chama API e este arquivo não publica tabela de resultado de modelo. O fake do harness não substitui essa tabela: a saída dele é checagem de encanamento. Empilhar as quatro guardas fica fora do desenho, porque misturaria a causa da rejeição.
