# Especificação: harness de avaliação do PR3

Métricas contra o gabarito, fora das guardas. O fake exercita o encanamento. Sem resultado de modelo e sem chamada de rede na suíte.

## Requisitos

- FR-1: CLI `evaluate-guards` e `python -m guardrails_study.evaluate`.
- FR-2: Uma estratégia ou `--strategy all` sobre o dataset v1. Saída: `records.jsonl` e `summary.json`.
- FR-3: Backend padrão é o fake, com cenários `always_valid`, `malformed_then_valid` e `subtle`.
- FR-4: Provedor ao vivo só com `--allow-live` e chave no ambiente. Sem os dois, código 2 e nenhum pedido.
- FR-5: Taxas de aceite, inválido que escapou, falso rejeição, exaustão de orçamento, média e percentis de tentativas e latência. Tokens e custo só quando existirem.
- FR-6: Quebra por estratégia e por categoria. Erro de transporte fica de fora do denominador.

## Critérios de aceite

- AC-1: Given o cenário `always_valid`, when o fake percorre as 45 tarefas nas quatro estratégias, then cada linha casa com o gabarito, as taxas de escape e falso rejeição são 0 e tokens ficam null.
- AC-2: Given registros montados à mão com um erro no meio, when a agregação roda, then o erro não entra na taxa e, se não houver linha ok, a taxa é null.
- AC-3: Given provedor `openai` ou `anthropic` sem `--allow-live` ou sem chave, when a CLI roda, then o código é 2, a mensagem não contém o segredo e nenhum cliente é montado.
- AC-4: Given o cenário `subtle`, when as guardas determinísticas e o revisor scriptado rodam, then as primeiras aceitam um JSON diferente do gabarito e o revisor scriptado esgota o orçamento sem falso rejeição.

## Casos de borda

- EC-1: JSON quebrado recusado não é escape nem falso rejeição.
- EC-2: Aceitar na última tentativa não esgota o orçamento.
- EC-3: Bool não equivale a 0 ou 1. `20` e `20.0` equivalem. Ordem de chave não conta.
- EC-4: Soma de tokens ou de custo é null se qualquer linha ok não tiver o campo.
- EC-5: `--scenario` com provedor ao vivo não envia pedido.
- EC-6: Revisor ao vivo sem flag só bloqueia quando a estratégia inclui `second_llm`. Gerador fake com revisor ao vivo também recusa, mesmo com o flag.

## Restrições

- Sem chamada de rede na suíte. Sem chave no repositório.
- Sem tabela de resultado de modelo no README.
- `valid` não muda de significado: é o aceite da guarda.

## Pontos de integração

- `RunResult` e `run_strategy`.
- `load_eval_tasks` e `data/eval/tasks.v1.jsonl`.
- `Settings.from_env` e `build_llm`, este só depois da porta do provedor ao vivo.
