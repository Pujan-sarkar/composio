import types
import typing as t
from inspect import Signature

from langchain_core.tools import StructuredTool

from composio.client.enums import Action, App, Tag
from composio.constants import DEFAULT_ENTITY_ID
from composio.sdk.shared_utils import (
    get_signature_format_from_schema_params,
    json_schema_to_model,
)
from composio.tools import ComposioToolSet as BaseComposioToolSet


class ComposioToolSet(BaseComposioToolSet):
    """
    Composio toolset for Langchain framework.

    Example:
    ```python
        import os
        import dotenv

        from composio_langchain import App, ComposioToolSet
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI

        from langchain import hub


        # Load environment variables from .env
        dotenv.load_dotenv()


        # Pull relevant agent model.
        prompt = hub.pull("hwchase17/openai-functions-agent")

        # Initialize tools.
        openai_client = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        composio_toolset = ComposioToolSet()

        # Get All the tools
        tools = composio_toolset.get_tools(apps=[App.GITHUB])

        # Define task
        task = "Star a repo SamparkAI/docs on GitHub"

        # Define agent
        agent = create_openai_functions_agent(openai_client, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Execute using agent_executor
        agent_executor.invoke({"input": task})
    ```
    """

    def __init__(
        self,
        api_key: t.Optional[str] = None,
        base_url: t.Optional[str] = None,
        entity_id: str = DEFAULT_ENTITY_ID,
    ) -> None:
        """
        Initialize composio toolset.

        :param api_key: Composio API key
        :param base_url: Base URL for the Composio API server
        :param entity_id: Entity ID for making function calls
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            runtime="langchain",
            entity_id=entity_id,
        )

    def _wrap_tool(self, schema: t.Dict[str, t.Any]) -> StructuredTool:
        """Wraps composio tool as Langchain StructuredTool object."""
        app = schema["appName"]
        action = schema["name"]
        description = schema["description"]

        def function(**kwargs: t.Any) -> t.Dict:
            f"""Wrapper function for {action}."""
            return self.execute_action(
                action=Action.from_app_and_action(
                    app=app,
                    name=action,
                ),
                params=kwargs,
            )

        parameters = json_schema_to_model(
            json_schema=schema["parameters"],
        )
        action_func = types.FunctionType(
            function.__code__,
            globals=globals(),
            name=action,
            closure=function.__closure__,
        )
        action_func.__signature__ = Signature(  # type: ignore
            parameters=get_signature_format_from_schema_params(
                schema_params=schema["parameters"]
            )
        )
        action_func.__doc__ = description
        return StructuredTool.from_function(
            name=action,
            description=description,
            args_schema=parameters,
            return_schema=True,
            func=action_func,
        )

    def get_actions(self, actions: t.Sequence[Action]) -> t.Sequence[StructuredTool]:
        """
        Get composio tools wrapped as Langchain StructuredTool objects.

        :param actions: List of actions to wrap
        :return: Composio tools wrapped as `StructuredTool` objects
        """

        return [
            self._wrap_tool(schema=tool.model_dump(exclude_none=True))
            for tool in self.client.actions.get(actions=actions)
        ]

    def get_tools(
        self,
        apps: t.Sequence[App],
        tags: t.Optional[t.List[t.Union[str, Tag]]] = None,
    ) -> t.Sequence[StructuredTool]:
        """
        Get composio tools wrapped as Langchain StructuredTool objects.

        :param apps: List of apps to wrap
        :param tags: Filter the apps by given tags
        :return: Composio tools wrapped as `StructuredTool` objects
        """

        return [
            self._wrap_tool(schema=tool.model_dump(exclude_none=True))
            for tool in self.client.actions.get(apps=apps, tags=tags)
        ]