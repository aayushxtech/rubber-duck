import sys

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static

try:
    from client import (
        BackendAPIError,
        BackendConnectionError,
        RubberDuckClient,
        RubberDuckError,
    )
    from session import Session
except ImportError:
    from cli.client import (
        BackendAPIError,
        BackendConnectionError,
        RubberDuckClient,
    )
    from cli.session import Session


class RubberDuckApp(App):
    """Minimal Textual interactive terminal application for RubberDuck."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #header_bar {
        dock: top;
        height: 3;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
        content-align: center middle;
        text-style: bold;
    }

    #conversation {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        background: $panel;
    }

    #status_bar {
        height: 1;
        padding: 0 1;
        color: $warning;
        background: $surface-darken-1;
    }

    #user_input {
        dock: bottom;
        margin: 0;
        border: solid $accent;
    }

    #hint_bar {
        dock: bottom;
        height: 1;
        background: $surface-darken-2;
        color: $text-muted;
        content-align: center middle;
    }
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__()
        self.client = RubberDuckClient(base_url=base_url)
        self.session = Session()
        self.is_busy = False
        self.backend_connected = False

    def compose(self) -> ComposeResult:
        yield Static("🦆 RubberDuck CLI — Connecting to backend...", id="header_bar")
        yield RichLog(id="conversation", wrap=True, markup=True)
        yield Static("Ready", id="status_bar")
        yield Static(
            "Commands: /ask <q> | /reply <r> | /analysis | /evidence | /status | /clear | /help | /quit",
            id="hint_bar",
        )
        yield Input(
            placeholder="Ask a question (/ask <q>), reply to Socratic inquiry, or type /help...",
            id="user_input",
        )

    async def on_mount(self) -> None:
        """Check backend connection and display initial greeting."""
        conv = self.query_one("#conversation", RichLog)
        conv.write("[bold cyan]Welcome to RubberDuck![/bold cyan]")
        conv.write("A critical thinking and Socratic reasoning assistant.\n")
        conv.write(
            "Type [bold yellow]/ask <your question>[/bold yellow] or enter your question directly below to begin."
        )
        conv.write(
            "Type [bold]/help[/bold] at any time for the list of available commands.\n"
        )

        await self._check_backend_status()

    async def _check_backend_status(self) -> bool:
        """Check health of backend and update the header."""
        self.backend_connected = await self.client.check_health()
        self._update_header()
        return self.backend_connected

    def _update_header(self) -> None:
        header = self.query_one("#header_bar", Static)
        status_color = "green" if self.backend_connected else "red"
        status_text = "Connected" if self.backend_connected else "Disconnected"

        session_state = "Idle"
        if self.session.is_awaiting_clarification:
            session_state = f"Awaiting Clarification (Turn {self.session.turn_count})"
        elif self.session.is_completed:
            session_state = f"Completed (Turns: {self.session.turn_count})"
        elif self.session.is_active:
            session_state = f"Active (Turn {self.session.turn_count})"

        header.update(
            f"🦆 [bold]RubberDuck[/bold] | "
            f"Backend: [{status_color}]{self.client.base_url} ({status_text})[/{status_color}] | "
            f"Session: [bold yellow]{session_state}[/bold yellow]"
        )

    def _set_status(self, text: str, is_loading: bool = False) -> None:
        bar = self.query_one("#status_bar", Static)
        if is_loading:
            bar.update(f"[bold yellow]⏳ {text}[/bold yellow]")
        else:
            bar.update(f"[dim]{text}[/dim]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command or message input."""
        raw_text = event.value.strip()
        event.input.value = ""

        if not raw_text:
            return

        if self.is_busy:
            conv = self.query_one("#conversation", RichLog)
            conv.write(
                "[bold red]Please wait for the current reasoning request to complete.[/bold red]"
            )
            return

        conv = self.query_one("#conversation", RichLog)

        # Dispatch command if starts with "/"
        if raw_text.startswith("/"):
            parts = raw_text.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/quit":
                self.exit()
                return
            elif cmd == "/clear":
                conv.clear()
                conv.write("[dim]Conversation cleared.[/dim]")
                return
            elif cmd == "/help":
                self._display_help()
                return
            elif cmd == "/status":
                await self._display_status()
                return
            elif cmd == "/analysis":
                self._display_analysis()
                return
            elif cmd == "/evidence":
                self._display_evidence()
                return
            elif cmd == "/ask":
                if not arg:
                    conv.write("[yellow]Usage: /ask <question>[/yellow]")
                    return
                self._handle_ask(arg)
                return
            elif cmd == "/reply":
                if not arg:
                    conv.write("[yellow]Usage: /reply <response>[/yellow]")
                    return
                if not self.session.is_awaiting_clarification:
                    conv.write(
                        "[yellow]RubberDuck is not awaiting clarification. Use /ask <question> to start a new inquiry.[/yellow]"
                    )
                    return
                self._handle_reply(arg)
                return
            else:
                conv.write(
                    f"[red]Unknown command: {cmd}. Type /help for available commands.[/red]"
                )
                return

        # Ordinary text input:
        if self.session.is_awaiting_clarification:
            self._handle_reply(raw_text)
        else:
            # When session is completed or idle, treat input as a new question
            self._handle_ask(raw_text)

    def _display_help(self) -> None:
        conv = self.query_one("#conversation", RichLog)
        conv.write("[bold cyan]Available Commands:[/bold cyan]")
        conv.write(
            "  [bold yellow]/ask <question>[/bold yellow]   — Start a new reasoning session"
        )
        conv.write(
            "  [bold yellow]/reply <text>[/bold yellow]     — Respond to a Socratic question"
        )
        conv.write(
            "  [bold yellow]/analysis[/bold yellow]         — Display the latest structured analysis"
        )
        conv.write(
            "  [bold yellow]/evidence[/bold yellow]         — Display collected search evidence"
        )
        conv.write(
            "  [bold yellow]/status[/bold yellow]           — Show backend connection and session status"
        )
        conv.write(
            "  [bold yellow]/clear[/bold yellow]            — Clear conversation history"
        )
        conv.write(
            "  [bold yellow]/help[/bold yellow]             — Show this help guide"
        )
        conv.write(
            "  [bold yellow]/quit[/bold yellow]             — Exit the application"
        )
        conv.write(
            "[dim]Note: When awaiting clarification, simply type your response without any prefix.[/dim]\n"
        )

    async def _display_status(self) -> None:
        conv = self.query_one("#conversation", RichLog)
        await self._check_backend_status()
        conn_str = (
            "[green]Connected[/green]"
            if self.backend_connected
            else "[red]Disconnected[/red]"
        )
        conv.write("[bold cyan]System & Session Status:[/bold cyan]")
        conv.write(f"  Backend URL: {self.client.base_url} ({conn_str})")
        conv.write(f"  Session Active: {self.session.is_active}")
        conv.write(f"  Session Completed: {self.session.is_completed}")
        conv.write(f"  Turn Count: {self.session.turn_count}")
        conv.write(
            f"  Awaiting Clarification: {self.session.is_awaiting_clarification}"
        )
        if self.session.initial_question:
            conv.write(
                f"  Original Question: [dim]{self.session.initial_question}[/dim]"
            )
        conv.write("")

    def _display_analysis(self) -> None:
        conv = self.query_one("#conversation", RichLog)
        analysis = self.session.latest_analysis
        if not analysis:
            conv.write("[yellow]No analysis available in the current session.[/yellow]")
            return

        conv.write("[bold cyan]════ Latest Analysis ════[/bold cyan]")
        conv.write(f"[bold]Summary:[/bold] {analysis.get('analysis', '(None)')}\n")

        assumptions = analysis.get("assumptions") or []
        conv.write(f"[bold]Assumptions ({len(assumptions)}):[/bold]")
        if assumptions:
            for idx, a in enumerate(assumptions, 1):
                conv.write(f"  {idx}. {a}")
        else:
            conv.write("  (No explicit assumptions listed)")

        uncertainties = analysis.get("uncertainties") or []
        conv.write(f"\n[bold]Uncertainties ({len(uncertainties)}):[/bold]")
        if uncertainties:
            for idx, u in enumerate(uncertainties, 1):
                conv.write(f"  {idx}. {u}")
        else:
            conv.write("  (No remaining uncertainties)")

        conv.write(
            f"\n[bold]External Research Needed:[/bold] {analysis.get('needs_search', False)}"
        )
        conv.write("[bold cyan]═════════════════════════[/bold cyan]\n")

    def _display_evidence(self) -> None:
        conv = self.query_one("#conversation", RichLog)
        evidence = self.session.latest_evidence
        if not evidence or not evidence.get("raw_results"):
            conv.write(
                "[yellow]No external evidence collected for this session.[/yellow]"
            )
            return

        results = evidence.get("raw_results", [])
        conv.write(
            f"[bold cyan]════ Collected Evidence ({len(results)} items) ════[/bold cyan]"
        )
        for idx, item in enumerate(results, 1):
            title = item.get("title") or item.get("query", "Evidence source")
            url = item.get("url", "N/A")
            content = item.get("content", "(No snippet available)")
            conv.write(f"[bold green][{idx}] {title}[/bold green]")
            conv.write(f"    [dim]URL: {url}[/dim]")
            conv.write(f"    Snippet: {content}\n")
        conv.write("[bold cyan]════════════════════════════════════════[/bold cyan]\n")

    @work(exclusive=True)
    async def _handle_ask(self, question: str) -> None:
        """Async worker to start reasoning on a question."""
        self.is_busy = True
        self._set_status(
            "Analyzing question and forming reasoning plan...", is_loading=True
        )
        conv = self.query_one("#conversation", RichLog)
        conv.write(f"\n[bold blue]👤 You:[/bold blue] {question}")

        self.session.start_new(question)
        self._update_header()

        try:
            data = await self.client.start_reasoning(question)
            self.backend_connected = True
            self.session.update_from_response(data)
            self._render_backend_response(data)
        except BackendConnectionError as e:
            self.backend_connected = False
            conv.write(f"[bold red]❌ Connection Error:[/bold red] {e}")
            self.session.reset()
        except BackendAPIError as e:
            conv.write(
                f"[bold red]❌ Backend Error ({e.status_code}):[/bold red] {e.detail}"
            )
            self.session.reset()
        except Exception as e:
            conv.write(f"[bold red]❌ Unexpected Error:[/bold red] {e}")
            self.session.reset()
        finally:
            self.is_busy = False
            self._set_status("Ready")
            self._update_header()

    @work(exclusive=True)
    async def _handle_reply(self, reply_text: str) -> None:
        """Async worker to send clarification reply."""
        self.is_busy = True
        self._set_status(
            "Processing clarification and continuing inquiry...", is_loading=True
        )
        conv = self.query_one("#conversation", RichLog)
        conv.write(f"\n[bold blue]👤 You (Reply):[/bold blue] {reply_text}")

        try:
            data = await self.client.reply_reasoning(self.session, reply_text)
            self.backend_connected = True
            self.session.update_from_response(data)
            self._render_backend_response(data)
        except BackendConnectionError as e:
            self.backend_connected = False
            conv.write(f"[bold red]❌ Connection Error:[/bold red] {e}")
        except BackendAPIError as e:
            conv.write(
                f"[bold red]❌ Backend Error ({e.status_code}):[/bold red] {e.detail}"
            )
        except Exception as e:
            conv.write(f"[bold red]❌ Unexpected Error:[/bold red] {e}")
        finally:
            self.is_busy = False
            self._set_status("Ready")
            self._update_header()

    def _render_backend_response(self, data: dict) -> None:
        """Render Socratic question or final answer in the conversation log."""
        conv = self.query_one("#conversation", RichLog)

        socratic_q = data.get("socratic_question")
        final_answer = data.get("final_answer")

        if socratic_q:
            conv.write(
                f"\n[bold yellow]🦆 RubberDuck (Socratic Question):[/bold yellow]\n{socratic_q}\n"
            )
            conv.write(
                "[dim](Type your response below to clarify, or use /analysis to view hidden assumptions)[/dim]"
            )

        if final_answer:
            conv.write(
                f"\n[bold green]🦆 RubberDuck (Final Conclusion):[/bold green]\n{final_answer}\n"
            )
            conv.write(
                "[dim](Inquiry complete. You can enter a new question or use /analysis or /evidence)[/dim]"
            )


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    app = RubberDuckApp(base_url=url)
    app.run()
