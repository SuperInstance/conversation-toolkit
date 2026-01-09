"""Tests for templates module."""

import pytest

from conversation_toolkit import (
    Role,
    Template,
    TemplateEngine,
    render_template,
    load_common_templates,
)


class TestTemplate:
    """Test Template class."""

    def test_create_template(self):
        """Test creating a template."""
        template = Template(
            role=Role.USER,
            content="Hello, {name}!",
            required_params=["name"]
        )
        assert template.role == Role.USER
        assert template.required_params == ["name"]

    def test_render_template(self):
        """Test rendering a template."""
        template = Template(
            role=Role.USER,
            content="Hello, {name}!"
        )
        msg = template.render(name="World")
        assert msg.content == "Hello, World!"
        assert msg.role == Role.USER

    def test_render_missing_required_param(self):
        """Test rendering with missing required parameter."""
        template = Template(
            role=Role.USER,
            content="Hello, {name}!",
            required_params=["name"]
        )
        with pytest.raises(ValueError, match="Missing required parameter"):
            template.render()

    def test_render_unexpected_param(self):
        """Test rendering with unexpected parameter."""
        template = Template(
            role=Role.USER,
            content="Hello, {name}!",
            parameters={"unexpected": None}
        )
        # This should raise if unexpected param validation is on
        with pytest.raises(ValueError, match="Unexpected parameter"):
            template.render(name="World")


class TestTemplateEngine:
    """Test TemplateEngine class."""

    def test_create_engine(self):
        """Test creating template engine."""
        engine = TemplateEngine()
        assert len(engine.templates) == 0
        assert len(engine.template_vars) == 0

    def test_register_template(self):
        """Test registering a template."""
        engine = TemplateEngine()
        template = engine.register(
            "greeting",
            Role.USER,
            "Hello, {name}!",
            parameters=["name"]
        )
        assert "greeting" in engine.templates
        assert template.required_params == ["name"]

    def test_render_registered_template(self):
        """Test rendering a registered template."""
        engine = TemplateEngine()
        engine.register(
            "greeting",
            Role.USER,
            "Hello, {name}!",
            parameters=["name"]
        )
        msg = engine.render("greeting", name="World")
        assert msg.content == "Hello, World!"

    def test_render_unknown_template(self):
        """Test rendering unknown template."""
        engine = TemplateEngine()
        with pytest.raises(ValueError, match="Unknown template"):
            engine.render("unknown")

    def test_set_var(self):
        """Test setting global template variable."""
        engine = TemplateEngine()
        engine.set_var("name", "World")
        engine.register(
            "greeting",
            Role.USER,
            "Hello, {name}!"
        )
        msg = engine.render("greeting")
        assert msg.content == "Hello, World!"

    def test_var_override(self):
        """Test that kwargs override template vars."""
        engine = TemplateEngine()
        engine.set_var("name", "World")
        engine.register(
            "greeting",
            Role.USER,
            "Hello, {name}!"
        )
        msg = engine.render("greeting", name="Universe")
        assert msg.content == "Hello, Universe!"

    def test_render_to_dict(self):
        """Test rendering template to dict."""
        engine = TemplateEngine()
        engine.register(
            "greeting",
            Role.USER,
            "Hello, {name}!",
            parameters=["name"]
        )
        result = engine.render_to_dict("greeting", name="World")
        assert result == {"role": "user", "content": "Hello, World!"}

    def test_get_template(self):
        """Test getting a registered template."""
        engine = TemplateEngine()
        engine.register("test", Role.USER, "Test")
        template = engine.get_template("test")
        assert template is not None
        assert template.content == "Test"

    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        engine = TemplateEngine()
        template = engine.get_template("nonexistent")
        assert template is None

    def test_list_templates(self):
        """Test listing all templates."""
        engine = TemplateEngine()
        engine.register("t1", Role.USER, "Test 1")
        engine.register("t2", Role.USER, "Test 2")
        templates = engine.list_templates()
        assert set(templates) == {"t1", "t2"}

    def test_load_from_dict(self):
        """Test loading templates from dict."""
        engine = TemplateEngine()
        templates = {
            "greeting": {
                "role": "user",
                "content": "Hello!",
                "parameters": []
            }
        }
        engine.load_from_dict(templates)
        assert "greeting" in engine.templates

    def test_load_from_json(self):
        """Test loading templates from JSON."""
        engine = TemplateEngine()
        json_str = '{"greeting": {"role": "user", "content": "Hello!", "parameters": []}}'
        engine.load_from_json(json_str)
        assert "greeting" in engine.templates


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_render_template(self):
        """Test render_template function."""
        msg = render_template("Hello, {name}!", name="World")
        assert msg.content == "Hello, World!"
        assert msg.role == Role.USER

    def test_render_template_with_role(self):
        """Test render_template with custom role."""
        msg = render_template(
            "You are {agent_role}.",
            role=Role.SYSTEM,
            agent_role="helper"
        )
        assert msg.role == Role.SYSTEM
        assert "helper" in msg.content

    def test_load_common_templates(self):
        """Test loading common templates."""
        engine = TemplateEngine()
        load_common_templates(engine)
        templates = engine.list_templates()
        assert "greeting" in templates
        assert "clarification" in templates
        assert "summarization" in templates
        assert "explain_code" in templates

    def test_common_template_greeting(self):
        """Test common greeting template."""
        engine = TemplateEngine()
        load_common_templates(engine)
        msg = engine.render("greeting")
        assert "help" in msg.content.lower()

    def test_common_template_clarification(self):
        """Test common clarification template."""
        engine = TemplateEngine()
        load_common_templates(engine)
        msg = engine.render("clarification", topic="AI")
        assert "AI" in msg.content
        assert "clarify" in msg.content.lower()

    def test_common_template_summarization(self):
        """Test common summarization template."""
        engine = TemplateEngine()
        load_common_templates(engine)
        text = "This is a long text to summarize."
        msg = engine.render("summarization", text=text)
        assert text in msg.content

    def test_common_template_explain_code(self):
        """Test common explain_code template."""
        engine = TemplateEngine()
        load_common_templates(engine)
        msg = engine.render("explain_code", code="print('hello')")
        assert "print('hello')" in msg.content
        assert "explain" in msg.content.lower()
