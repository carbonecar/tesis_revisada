# EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
Official implementation of this ACL 2024 paper.

It's based on [Foundation](https://github.com/MaciejMacko/ai-economist), An Economic Simulation Framework, which is announced by this paper: 

Zheng, Stephan, et al. "The ai economist: Improving equality and productivity with ai-driven tax policies." arXiv preprint arXiv:2004.13332 (2020).

# Run
Simulate with GPT-3.5, 100 agents, and 240 months (fill openai.api_key in simulate_utils.py): 

`python simulate.py --policy_model gpt --num_agents 100 --episode_length 240`

Simulate with Composite, 100 agents, and 240 months:

`python simulate.py --policy_model complex --num_agents 100 --episode_length 240`

For RL approaches, *i.e.*, **The ai economist**, we just follow their training codes and use the trained models for simulations. See appendix in the paper for details.

# Tests
`pytest tests/` (debe correrse desde esta carpeta, porque `simulate.py` abre `config.yaml` con ruta relativa).

Cubre `extract_action` (`simulate.py`), la función que parsea la respuesta del LLM a `[work, consumption]` — incluye casos reales de modelos que envuelven el JSON en explicaciones, code fences ```json``` o comillas tipográficas.

# Habilitar acceso a modelos Anthropic en Bedrock

Para usar `ECON_BACKEND=bedrock` con un modelo Anthropic nuevo (p. ej. Claude Sonnet 4.6/5), la cuenta AWS necesita aceptar el acuerdo de uso (EULA/pricing) de ese modelo antes de poder invocarlo — da `AccessDeniedException` / `ResourceNotFoundException` hasta hacerlo. Pasos (reemplazar `<MODEL_ID>` por el id sin el prefijo de región, p. ej. `anthropic.claude-sonnet-4-6`):

```bash
# 1. Ver qué modelos/perfiles de inferencia están en el catálogo de la cuenta
aws bedrock list-foundation-models --region us-east-1 \
  --query "modelSummaries[?contains(modelId, 'sonnet')].[modelId, modelLifecycle.status]" --output table

aws bedrock list-inference-profiles --region us-east-1 \
  --query "inferenceProfileSummaries[*].inferenceProfileId" --output text | tr '\t' '\n' | grep -i sonnet

# 2. Ver estado de acceso actual del modelo
aws bedrock get-foundation-model-availability --region us-east-1 --model-id <MODEL_ID>

# 3. Listar la oferta de EULA/pricing pendiente y aceptarla
TOKEN=$(aws bedrock list-foundation-model-agreement-offers --region us-east-1 \
  --model-id <MODEL_ID> --query "offers[0].offerToken" --output text)

aws bedrock create-foundation-model-agreement --region us-east-1 \
  --model-id <MODEL_ID> --offer-token "$TOKEN"

# 4. Probar la invocación (usar el id CON prefijo de región si el modelo requiere inference profile)
aws bedrock-runtime converse --region us-east-1 --model-id us.<MODEL_ID> \
  --messages '[{"role":"user","content":[{"text":"di OK"}]}]'
```

Nota: para modelos Anthropic, además hay que completar **una sola vez por cuenta** el formulario "use case" (lo pide la consola de Bedrock al seleccionar un modelo Anthropic por primera vez, o vía `aws bedrock put-use-case-for-model-access`). Si un modelo devuelve `AccessDeniedException` mencionando "contact AWS Sales" incluso después de aceptar el agreement, es un rollout restringido de ese modelo puntual — no se resuelve con estos comandos, hay que esperar a que AWS/Anthropic habiliten la cuenta o contactar a AWS Sales.

Fuente: [AWS Bedrock — Request access to models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#model-access-permissions)

# Update in 2024.8.16
The simulation was only tested using gpt-3.5-turbo-0613, but this model seems to no longer be accessible and has been replaced by gpt-4o-mini. If `gpt_error` is significantly greater than 0 (e.g., exceeding 10), meaning GPT generates many unreasonable decisions, please adjust the prompts accordingly, especially the parts related to format instruction:

*"Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods)."*
