# ============================================================
# COACHLENS
# ESTILO GLOBAL DE GRAFICAS PLOTLY
# ============================================================

COACH_NAVY = "#061E39"
COACH_NAVY_DEEP = "#002744"
COACH_SECONDARY = "#40536C"
COACH_YELLOW = "#FFDA38"
COACH_RED = "#D82030"
COACH_CREAM = "#FFF9E8"
COACH_GRID = "rgba(6,30,57,0.14)"

BEFORE_COLOR = "#FFDA38"
AFTER_COLOR = "#12366A"


def apply_coachlens_chart_style(
    fig,
    height=340,
    legend=False,
    x_title="",
    y_title="",
):
    """
    Estilo visual común para las gráficas de CoachLens.
    """

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=75,
            r=75,
            t=35,
            b=65,
        ),
        font=dict(
            family="Arial, sans-serif",
            color=COACH_NAVY,
            size=14,
        ),
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=legend,
    )

    fig.update_xaxes(
        tickfont=dict(
            color=COACH_NAVY,
            size=14,
        ),
        title_font=dict(
            color=COACH_SECONDARY,
            size=14,
        ),
        gridcolor=COACH_GRID,
        linecolor="rgba(6,30,57,0.35)",
        tickcolor=COACH_NAVY,
        zeroline=False,
        automargin=True,
    )

    fig.update_yaxes(
        tickfont=dict(
            color=COACH_NAVY,
            size=14,
        ),
        title_font=dict(
            color=COACH_SECONDARY,
            size=14,
        ),
        gridcolor=COACH_GRID,
        linecolor="rgba(6,30,57,0.35)",
        tickcolor=COACH_NAVY,
        zeroline=False,
        automargin=True,
    )

    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(
                    color=COACH_NAVY,
                    size=13,
                ),
                bgcolor="rgba(0,0,0,0)",
            )
        )

    return fig