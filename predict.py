#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/3 0:49
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import typer

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()
