try: from numpy import linspace, meshgrid
except: pass
try: from sympy import lambdify, Symbol, Expr
except: pass
try: from plotly.graph_objects import Surface, Figure
except: pass
try: from random import choices
except: pass
try: from string import ascii_letters, digits
except: pass

class Plot3D:
    def __init__(self, expressions, x_min = -3, x_max = 3, y_min = -3, y_max = 3, precision = 100):
        try:
            iter(expressions)
            funcs = [z for z in expressions]
        except TypeError:
            funcs = [expressions]    

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.X, self.Y = meshgrid(linspace(x_min, x_max, precision), linspace(y_min, y_max, precision))

        evaluated = [*map(lambda f: f((self.X, self.Y)), funcs)]
        self.figure = self.generateoFigures(evaluated)
        self.show()

    def show(self):
        self.figure.show()

    def generateoFigures(self, Z):
        surfaces = []
        for F in Z:
            surfaces.append(
                Surface(x = self.X, y = self.Y, z = F, colorscale = "Viridis", showscale = True)
            )

        range_start, range_end = [min(self.x_min, self.y_min) - 2, max(self.x_max, self.y_max) + 2]
    
        fig = Figure(data = surfaces)
        fig.update_layout(
            scene = dict(
                xaxis = dict(nticks=10, range=[range_start, range_end],),
                yaxis = dict(nticks=10, range=[range_start, range_end],),
                zaxis = dict(nticks=10, range=[range_start, range_end],),
            ),
            width=1000,
            height = 600,
            margin=dict(r=20, l=10, b=10, t=10))
        return fig

    @staticmethod
    def evaluateSympyExpr(f, X, Y):
        assert isinstance(f, Expr), f"\n\nError, {str(f)} is not a Sympy expression. Try to use template: \n'from sympy import symbols\nx, y = symbols(\'x y\')\nf = x * y # sympy expression\nPlot3D(f)\n\n"
        variables = list(f.free_symbols)
        assert len(variables) <= 2, f"\n\nError, expression {str(f)} contains more than 2 variables, 'plotly' library can't plot multi-dimension functions\n\n"

        if len(variables) == 1:
            variables.append(Symbol(''.join(choices(ascii_letters, k = 1))))
        elif len(variables) == 0:
            variables.append(Symbol(''.join(choices(ascii_letters, k = 1))))
            variables.append(Symbol(''.join(choices(ascii_letters, k = 1))))
        
        return lambdify(variables, f, "numpy")(X, Y)