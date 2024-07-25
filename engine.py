import math

class Value:
    """Defines a Value object, which stores a single scalar value and its gradient.

    When a primitive operation is called on this Value object (e.g. add/multiply/pow),
    a new Value object is returned which:
        1) keeps track of its child value objects
        2) has a "._backward" method defined which computes the vector-Jacobian
           product of the primative operation which created it

    After computing a series of operations with Value objects, one can call the
    ".backward" method. This recursively backpropagates (applies the chain rule)
    through the entire computational graph, accumulating gradients in the leaf
    Values of the graph.

    Your tasks:

        1) Implement the primitive operations which currently raise a NotImplementedError
        HINT: for each primitive operation, make sure to define its vector-Jacobian
        product i.e. return a new value object which has a "._backward" defined
        Because we are only dealing with scalar primitive operations, the
        vector-Jacobian product just reduces to two scalar values multiplied
        together (no matrix operations are needed)

        2) Implement the ".backward" method.
        HINT: In order to apply the chain rule properly, we need to call the
        ._backward method of each Value in the graph in topological order.
        So you must first sort the Values in the graph into this order and
        then apply the chain rule to this sorted list.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward

        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return (-self) + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return (self**-1) * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"



if __name__ == "__main__":

    # Run some tests on the value class

    x = Value(1)
    y = x + 2 - 3
    y.backward()
    assert x.grad == 1

    x = Value(1)
    y = 4*x/2
    y.backward()
    assert x.grad == 2

    x = Value(2)
    y = x**2
    y.backward()
    assert x.grad == 2*2

    x = Value(0)
    y = x.cos()
    y.backward()
    assert x.grad == 0

    x = Value(0)
    y = x.sin()
    y.backward()
    assert x.grad == 1

    x = Value(0)
    y = x*x.cos() + x**2 + 3
    y.backward()
    assert x.grad == 1




