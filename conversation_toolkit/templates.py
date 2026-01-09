"""
Template engine for message generation.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from string import Template as StringTemplate

from .conversation import Message, Role


@dataclass
class Template:
    """
    A message template.

    Example:
        template = Template(
            role=Role.USER,
            content="Explain {topic} in {style} tone"
        )
        rendered = template.render(topic="AI", style="formal")
    """
    role: Role
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)

    def render(self, **kwargs) -> Message:
        """Render template with parameters."""
        # Validate required params
        for param in self.required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate parameters match those expected
        for param in self.parameters:
            if param not in kwargs and param not in self.required_params:
                raise ValueError(f"Unexpected parameter: {param}")

        content = self.content.format(**kwargs)
        return Message(role=self.role, content=content)


class TemplateEngine:
    """
    Engine for managing and rendering templates.

    Supports multiple template formats including:
    - String templates (str.format)
    - JSON templates
    - Conditional templates
    - Loop templates

    Example:
        engine = TemplateEngine()

        # Register template
        engine.register("greeting", Role.USER, "Hello {name}!")

        # Render
        msg = engine.render("greeting", name="World")
    """

    def __init__(self):
        """Initialize the template engine."""
        self.templates: Dict[str, Template] = {}
        self.template_vars: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        role: Role,
        content: str,
        parameters: Optional[List[str]] = None
    ) -> Template:
        """
        Register a template.

        Args:
            name: Template name
            role: Message role
            content: Template content
            parameters: Optional list of expected parameters

        Returns:
            Created Template
        """
        template = Template(
            role=role,
            content=content,
            parameters={p: None for p in parameters or []},
            required_params=parameters or []
        )
        self.templates[name] = template
        return template

    def render(
        self,
        template_name: str,
        **kwargs
    ) -> Message:
        """
        Render a registered template.

        Args:
            template_name: Template name
            **kwargs: Template parameters

        Returns:
            Rendered Message
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        # Apply template vars as defaults
        params = {**self.template_vars, **kwargs}
        return self.templates[template_name].render(**params)

    def render_to_dict(
        self,
        template_name: str,
        **kwargs
    ) -> Dict[str, str]:
        """Render template and return as dict."""
        msg = self.render(template_name, **kwargs)
        return msg.to_dict()

    def set_var(self, key: str, value: Any) -> None:
        """Set a global template variable."""
        self.template_vars[key] = value

    def get_template(self, template_name: str) -> Optional[Template]:
        """Get a registered template."""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self.templates.keys())

    def load_from_dict(self, templates: Dict[str, Dict[str, Any]]) -> None:
        """
        Load templates from dictionary.

        Args:
            templates: Dict of {name: {role, content, parameters}}
        """
        for name, config in templates.items():
            self.register(
                name=name,
                role=Role(config["role"]),
                content=config["content"],
                parameters=config.get("parameters")
            )

    def load_from_json(self, json_str: str) -> None:
        """Load templates from JSON string."""
        data = json.loads(json_str)
        self.load_from_dict(data)


# Convenience functions

def render_template(
    content: str,
    role: Role = Role.USER,
    **kwargs
) -> Message:
    """
    Render a template string.

    Args:
        content: Template content with {placeholders}
        role: Message role
        **kwargs: Template parameters

    Returns:
        Rendered Message

    Example:
        msg = render_template("Hello {name}!", name="World")
    """
    template = Template(role=role, content=content)
    return template.render(**kwargs)


# Common templates

COMMON_TEMPLATES = {
    "greeting": {
        "role": "user",
        "content": "Hello! How can I help you today?",
        "parameters": []
    },
    "clarification": {
        "role": "user",
        "content": "Could you please clarify what you mean by {topic}?",
        "parameters": ["topic"]
    },
    "summarization": {
        "role": "user",
        "content": "Please summarize the following: {text}",
        "parameters": ["text"]
    },
    "explain_code": {
        "role": "user",
        "content": "Explain what this code does:\n```\n{code}\n```",
        "parameters": ["code"]
    }
}


def load_common_templates(engine: TemplateEngine) -> None:
    """Load common templates into an engine."""
    engine.load_from_dict(COMMON_TEMPLATES)
