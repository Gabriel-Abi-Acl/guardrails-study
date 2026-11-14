# Especificação: scaffold do PR1

Instrumentos para comparar guardas de saída estruturada. Sem métrica de modelo e sem chamada de rede na suíte.

## Requisitos

- FR-1: Pacote Python 3.11+ gerido por uv, com ruff e pytest.
- FR-2: CI executa lint e testes e fixa `astral-sh/setup-uv@v10.3.3`.
- FR-3: Cliente de LLM com fake de fila (`valid`, `malformed`, `subtle` e texto arbitrário). SDKs reais só no primeiro `complete` ao vivo.
- FR-4: Quatro guardas independentes: JSON Schema, Pydantic, regras, segundo LLM.
- FR-5: Runner com orçamento. Registro com saída final, `valid`, tentativas, guarda que rejeitou, latência, tokens e custo quando existirem.
- FR-6: Fixture sintética de 10 tarefas com schema, regras e gabarito.
- FR-7: Configuração por ambiente, com `.env.example` versionado e `.env` ignorado. Nenhuma chave no repositório.

## Critérios de aceite

- AC-1: Given o gabarito de cada tarefa, when as três guardas determinísticas rodam, then todas aceitam.
- AC-2: Given o exemplo `malformed`, when as guardas determinísticas rodam, then todas rejeitam e `json.loads` falha.
- AC-3: Given o exemplo `subtle`, when as guardas determinísticas rodam, then todas aceitam e o JSON difere do gabarito.
- AC-4: Given a data `2024-02-31`, when as guardas rodam, then o schema aceita e Pydantic e regras rejeitam.
- AC-5: Given total de linha incoerente ou SLA alto com prioridade `high`, when as guardas rodam, then só as regras rejeitam.
- AC-6: Given três saídas inválidas e orçamento 3, when o runner executa, then há 3 tentativas, `valid` é falso e a guarda rejeitadora fica no registro.
- AC-7: Given sucesso na segunda tentativa, when o runner executa, then para em 2 e o reparo cita o motivo, sem o gabarito.
- AC-8: Given uso completo e preços configurados, when o runner soma, then o custo é o produto informado. Sem uso ou sem preço, tokens ou custo ficam null.
- AC-9: Given o revisor fora do contrato, when a guarda `second_llm` avalia, then rejeita com `checker_malformed`.
- AC-10: Given import do pacote, when a suíte olha `sys.modules`, then `openai` e `anthropic` não foram importados.

## Casos de borda

- EC-1: Fila do fake vazia levanta erro. Não há texto inventado.
- EC-2: Campo extra passa nas regras listadas e falha em schema e Pydantic.
- EC-3: `"1990"` e `"false"` não são coeridos para inteiro nem booleano.
- EC-4: `max_attempts < 1` levanta `ValueError`. A estratégia `json_schema` usa 1 tentativa mesmo com `MAX_ATTEMPTS=3`.
- EC-5: Chave de API não entra no `repr` do settings nem do cliente, nem na mensagem de credencial ausente.
- EC-6: Revisor sem contagem de tokens anula o uso da tentativa inteira, mesmo que o gerador tenha informado tokens.

## Restrições

- Tarefas sintéticas e públicas. Sem dado pessoal real.
- Sem empilhar guardas neste PR.
- Sem resultado experimental no README.
- Adaptadores ao vivo ficam fora da rede na suíte. `temperature=0`.

## Pontos de integração

- Fixture `guardrails_study/fixtures/tasks.json`, lida pelo pacote.
- Variáveis em `.env.example`.
- Workflow `.github/workflows/ci.yml`.
- PR3 deve consumir `RunResult` sem redefinir o significado de `valid`.
