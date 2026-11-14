"""
Handoff functions.
They'll specify a control transfer between agents.
Construct the requested agent for the supplied user_id.
"""

from typing import Any


def transfer_to_general_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of GeneralAgent.

    :Paragraph user_id: User Identifier.
    :param openai_api_key: API key OpenAI.
    :Param knowledge_client: Knowledge Base Client (WeaviateClient)).
    :Param async_client: Basic URL OpenAI API.
    :return: GeneralAgent.
    """
    from mas.agents.general_agent import GeneralAgent

    return GeneralAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_ui_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of UIAgent.

    :Paragraph user_id: User Identifier.
    :param openai_api_key: API key OpenAI.
    :param knowledge_client: (Operate) Knowledge Base Client.
    :Param async_client: (Operate) Basic URL OpenAI API.
    :return: UIAgent.
    """
    from mas.agents.ui_agent import UIAgent

    return UIAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_wp_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of WP_Agent.

    :Paragraph user_id: User Identifier.
    :param openai_api_key: API key OpenAI.
    :param knowledge_client: (Operate) Knowledge Base Client.
    :Param async_client: (Operate) Basic URL OpenAI API.
    :return: WP_Agent.
    """
    from mas.agents.wp_agent import WP_Agent

    return WP_Agent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_conflict_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of ConflictAgent.

    :Paragraph user_id: User Identifier.
    :param openai_api_key: API key OpenAI.
    :Param async_client: (Operate) Basic URL OpenAI API.
    :return: ConflictAgent.
    """
    from mas.agents.conflict_agent import ConflictAgent

    return ConflictAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_october_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of OctoberAgent.

    :Paragraph user_id: User Identifier.
    :param openai_api_key: API key OpenAI.
    :Param knowledge_client: Knowledge client.
    :Param async_client: (Operate) Basic URL OpenAI API.
    :return: OctoberAgent.
    """
    from mas.agents.october_agent import OctoberAgent

    return OctoberAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_services_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of ServiceAgent.
    """
    from mas.agents.services_agent import ServicesAgent

    return ServicesAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_consultation_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of ConsultationAgent.
    """
    from mas.agents.consultation_agent import ConsultationAgent

    return ConsultationAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_blog_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of BlogAgent.
    """
    from mas.agents.blog_agent import BlogAgent

    return BlogAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_onboarding_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of OnboardingAgent.
    """
    from mas.agents.onboarding_agent import OnboardingAgent

    return OnboardingAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_emotion_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of EmotionAgent.
    """
    from mas.agents.emotion_agent import EmotionAgent

    return EmotionAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_to_sales_agent(
    user_id: str,
    openai_api_key: str,
    knowledge_client: Any,
    async_client: Any,
    first_name: str = "",
    last_name: str = "",
    username: str = "",
    about_me: str = "",
) -> Any:
    """
    Return an instance of SalesAgent.
    """
    from mas.agents.sales_agent import SalesAgent

    return SalesAgent(
        user_id=user_id,
        openai_api_key=openai_api_key,
        knowledge_client=knowledge_client,
        async_client=async_client,
        first_name=first_name,
        last_name=last_name,
        username=username,
        about_me=about_me,
    )


def transfer_back_to_orchestrator(user_id: str) -> None:
    """
    The control-back function of the Orchestrator Agent.

    :Paragraph user_id: User Identifier.
    """
    print(f"[Handoff] Returning Orchestrator's control for {user_id}")
