# guardrails-study

Estudo técnico sobre como validar saída estruturada de modelos de linguagem. O recorte deste repositório é instrumental: quatro estratégias de guarda, um cliente de LLM agnóstico de provedor e um runner com orçamento de tentativas. As tarefas são sintéticas e públicas. Este repositório, no PR1, não contém resultado experimental.

## Objetivo

Comparar quatro formas de decidir se a saída de um LLM pode ser aceita como JSON de uma tarefa de extração ou formatação:

1. validação somente por JSON Schema;
2. validação por modelo Pydantic, com nova pergunta quando a saída falha;
3. regras determinísticas (expressão regular e invariantes de negócio);
4. um segundo LLM no papel de revisor.

O que se quer medir nos PRs seguintes é a taxa de saída inválida que a guarda aceita, a taxa de falso rejeição contra o gabarito, o número de tentativas, a latência e o custo extra de tokens quando o provedor informa uso. PR1 constrói esses registros e os testes offline. Não mede um modelo real.

`valid` no registro significa "a guarda aceitou a saída final". Não significa "bate com o gabarito". Concordância com o gabarito é métrica do PR3, calculada fora da guarda.

## Hipótese

Falhas de saída estruturada não são um único fenômeno.

- JSON Schema e Pydantic devem barrar JSON quebrado, tipo errado e valor fora de enum ou padrão. Não devem, sozinhos, perceber que um JSON bem formado extraiu o valor errado do texto.
- O modelo Pydantic deste estudo acompanha o schema nas restrições estruturais, recusa coerção (`"1990"` não vira inteiro, `"false"` não vira booleano) e ainda interpreta data como calendário real. A diferença operacional planejada da estratégia 2 é o re-ask: a guarda devolve o motivo e o runner pergunta de novo, até um orçamento.
- Regras determinísticas devem pegar invariantes que o schema do estudo deixa de fora de propósito: data impossível além do formato `YYYY-MM-DD`, produto de campos e teto condicional de SLA. A hipótese é que o falso rejeição delas só ocorre quando a regra está errada ou estreita demais. Isso ainda não foi medido.
- Um segundo LLM pode rejeitar extração infiel ao texto-fonte mesmo quando o JSON passa nas checagens estruturais. A hipótese é que isso reduz a taxa de inválidos que escapam e aumenta tokens, latência e falso rejeição. A direção e o tamanho desses efeitos são perguntas empíricas, não conclusão.

O re-ask só ajuda se o motivo devolvido for acionável e se ainda houver tentativas. Esgotar o orçamento conta como saída não aceita pela guarda.

## Metodologia planejada

Unidade experimental, a partir do PR2: uma tarefa sintética com instrução, texto-fonte, JSON Schema, modelo Pydantic, lista de regras e gabarito rotulado. PR1 fixa um placeholder de 10 tarefas para os testes. Nenhuma tarefa usa dado pessoal real. E-mails ficam em `example.com`. Identificadores são fictícios (`INV-1042`, `ORD-77`, `TRK-9`).

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

Métricas que o PR3 deve calcular fora das guardas, sempre contra o gabarito, e que este PR não reporta:

- inválido que escapou: a guarda aceitou e o JSON parseado difere do gabarito;
- falso rejeição: a guarda rejeitou e a saída era o gabarito;
- tentativas, latência agregada, tokens e custo quando existirem.

`examples.subtle`, na fixture atual, existe para o fake emitir o caso em que essas métricas divergem da decisão da guarda. Não é uma taxa.

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

## Roadmap

- PR2: dataset completo e gabarito revisado, no lugar do placeholder de 10 tarefas.
- PR3: avaliação e métricas (escapa, falso rejeição, tentativas, latência, tokens e custo) sobre o runner deste PR.
- PR4: experimentos com modelos reais e relatório. Sem números antes da execução registrada.

## O que este PR não faz

Não chama API, não estima qualidade de modelo e não publica tabela de resultado. Empilhar as quatro guardas fica fora do desenho, porque misturaria a causa da rejeição.
