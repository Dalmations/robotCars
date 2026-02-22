# car_tools/draw_debug.py
"""
Turtle debug UI (Windows-safe):
1) Draw obstacles by click+drag
2) ENTER toggles to goal-placement mode
3) Click once to place goal (star)
4) P prints export JSON (segments + goal)
Optional:
- C clears
- SPACE finishes (closes window so your program can continue)
"""

from __future__ import annotations

import json
import turtle
from typing import List, Optional, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


class TurtleObstacleGoalDrawer:
    def __init__(
        self,
        world_size: Tuple[int, int] = (50, 50),
        pixels_per_cell: int = 12,
        title: str = "Draw obstacles, then place goal",
    ):
        self.world_w, self.world_h = world_size
        self.pixels_per_cell = pixels_per_cell

        self.segments: List[Segment] = []
        self.goal: Optional[Point] = None
        self.mode: str = "obstacles"  # "obstacles" or "goal"

        self._drawing: bool = False
        self._last: Optional[Point] = None
        self._done: bool = False

        # Screen setup
        self.screen = turtle.Screen()
        self.screen.title(title)

        px_w = self.world_w * pixels_per_cell
        px_h = self.world_h * pixels_per_cell
        self.screen.setup(width=px_w + 80, height=px_h + 140)

        # World coordinates (0..W, 0..H)
        self.screen.setworldcoordinates(0, 0, float(self.world_w), float(self.world_h))

        # Faster drawing
        self.screen.tracer(0, 0)

        # Pen turtle
        self.pen = turtle.Turtle(visible=False)
        self.pen.speed(0)
        self.pen.pensize(2)
        self.pen.color("black")
        self.pen.penup()

        # Goal turtle
        self.goal_t = turtle.Turtle(visible=False)
        self.goal_t.speed(0)
        self.goal_t.color("gold")
        self.goal_t.penup()

        # UI text
        self.ui = turtle.Turtle(visible=False)
        self.ui.speed(0)
        self.ui.penup()
        self.ui.color("gray25")

        self._draw_grid()
        self._set_instructions()
        self.screen.update()

        # Canvas + bindings
        self.canvas = self.screen.getcanvas()
        self._bind_canvas_mouse_events()
        self._bind_canvas_keys()

        self.screen.listen()  # still useful on some setups

    def run(self) -> Tuple[List[Segment], Optional[Point]]:
        self._pump()
        turtle.mainloop()
        return self.segments, self.goal

    # ---------------------------
    # Pixel -> World conversion (NO private turtle attrs)
    # ---------------------------
    def _event_to_world(self, event) -> Point:
        """
        Tk event.x/event.y are pixels relative to the canvas.
        With setworldcoordinates(0,0,W,H), we map:
          x_world = (x_px / canvas_width)  * W
          y_world = ((canvas_height - y_px) / canvas_height) * H
        """
        w_px = max(1, int(self.canvas.winfo_width()))
        h_px = max(1, int(self.canvas.winfo_height()))

        x_world = (event.x / w_px) * float(self.world_w)
        y_world = ((h_px - event.y) / h_px) * float(self.world_h)

        # clamp
        x_world = max(0.0, min(float(self.world_w), x_world))
        y_world = max(0.0, min(float(self.world_h), y_world))
        return (x_world, y_world)

    def world_to_grid(self, p: Point) -> Tuple[int, int]:
        x, y = p
        gx = int(round(x))
        gy = int(round(y))
        gx = max(0, min(self.world_w - 1, gx))
        gy = max(0, min(self.world_h - 1, gy))
        return gx, gy

    # ---------------------------
    # UI drawing
    # ---------------------------
    def _set_instructions(self) -> None:
        self.ui.clear()
        self.ui.goto(0.5, self.world_h + 0.8)
        if self.mode == "obstacles":
            msg = "Draw obstacles: click+drag. ENTER=goal mode | P=export | C=clear | SPACE=finish"
        else:
            msg = "Place goal: click once (star). ENTER=draw mode | P=export | C=clear | SPACE=finish"
        self.ui.write(msg, font=("Arial", 12, "normal"))

    def _draw_grid(self) -> None:
        g = turtle.Turtle(visible=False)
        g.speed(0)
        g.pensize(1)
        g.color("gray85")
        g.penup()

        step = 5
        for x in range(0, self.world_w + 1, step):
            g.goto(x, 0)
            g.pendown()
            g.goto(x, self.world_h)
            g.penup()

        for y in range(0, self.world_h + 1, step):
            g.goto(0, y)
            g.pendown()
            g.goto(self.world_w, y)
            g.penup()

    def _draw_star(self, x: float, y: float, size: float = 1.4) -> None:
        self.goal_t.clear()
        self.goal_t.penup()
        self.goal_t.goto(x, y)
        self.goal_t.setheading(90)
        self.goal_t.pendown()
        self.goal_t.begin_fill()
        for _ in range(5):
            self.goal_t.forward(size)
            self.goal_t.right(144)
        self.goal_t.end_fill()
        self.goal_t.penup()

    # ---------------------------
    # Canvas bindings
    # ---------------------------
    def _bind_canvas_mouse_events(self) -> None:
        def on_press(event):
            x, y = self._event_to_world(event)

            if self.mode == "goal":
                self.goal = (x, y)
                self._draw_star(x, y)
                self._set_instructions()
                self.screen.update()
                return

            # obstacle mode: start stroke
            self._drawing = True
            self._last = (x, y)
            self.pen.penup()
            self.pen.goto(x, y)
            self.screen.update()

        def on_motion(event):
            if self.mode != "obstacles" or not self._drawing or self._last is None:
                return

            x, y = self._event_to_world(event)
            cur = (x, y)

            # draw
            self.pen.pendown()
            self.pen.goto(x, y)
            self.pen.penup()

            # record
            self.segments.append((self._last, cur))
            self._last = cur

            self.screen.update()

        def on_release(_event):
            self._drawing = False
            self._last = None

        self.canvas.bind("<ButtonPress-1>", on_press)
        self.canvas.bind("<B1-Motion>", on_motion)
        self.canvas.bind("<ButtonRelease-1>", on_release)

        # Force focus so keys work on Windows
        try:
            self.canvas.focus_force()
        except Exception:
            pass

    def _bind_canvas_keys(self) -> None:
        def on_p(_e): self._print_export()
        def on_c(_e): self._clear_all()
        def on_space(_e): self._finish()
        def on_enter(_e): self._toggle_mode()

        self.canvas.bind_all("<KeyPress-p>", on_p)
        self.canvas.bind_all("<KeyPress-P>", on_p)
        self.canvas.bind_all("<KeyPress-c>", on_c)
        self.canvas.bind_all("<KeyPress-C>", on_c)
        self.canvas.bind_all("<space>", on_space)
        self.canvas.bind_all("<Return>", on_enter)

    # ---------------------------
    # Actions
    # ---------------------------
    def _toggle_mode(self) -> None:
        self.mode = "goal" if self.mode == "obstacles" else "obstacles"
        self._drawing = False
        self._last = None
        self._set_instructions()
        self.screen.update()

    def _clear_all(self) -> None:
        self.segments.clear()
        self.goal = None
        self.mode = "obstacles"
        self._drawing = False
        self._last = None
        self.pen.clear()
        self.goal_t.clear()
        self._set_instructions()
        self.screen.update()

    def _print_export(self) -> None:
        payload = {
            "world_size": [self.world_w, self.world_h],
            "segments": self.segments,
            "goal": self.goal,
        }
        print("\n=== Turtle Debug Export ===")
        print(f"Segments: {len(self.segments)}")
        print(f"Goal: {self.goal}")
        if self.goal is not None:
            print(f"Goal grid: {self.world_to_grid(self.goal)}")
        print("JSON:")
        print(json.dumps(payload))
        print("===========================\n")

    def _finish(self) -> None:
        self._done = True
        try:
            self.screen.bye()
        except turtle.Terminator:
            pass

    def _pump(self) -> None:
        if not self._done:
            self.screen.ontimer(self._pump, 50)
