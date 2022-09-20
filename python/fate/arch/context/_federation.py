from typing import Any, Callable, List, Literal, Optional, Tuple, TypeVar

from fate.interface import Parties as PartiesInterface
from fate.interface import Party as PartyInterface
from fate.interface import Future as FutureInterface
from fate.interface import Futures as FuturesInterface
from fate.interface import FPTensor, PHETensor, PHEEncryptor

from ..common import Party as PartyMeta
from ..federation.transfer_variable import IterationGC
from ..session import get_session
from ._namespace import Namespace


class FederationDeserializer:
    def do_deserialize(self, ctx, party):
        ...

    @classmethod
    def make_frac_key(cls, base_key, frac_key):
        return f"{base_key}__frac__{frac_key}"


T = TypeVar("T")


class Future(FutureInterface):
    def __init__(self, inside) -> None:
        self._inside = inside

    def unwrap_tensor(self) -> "FPTensor":

        assert isinstance(self._inside, FPTensor)
        return self._inside

    def unwrap_phe_encryptor(self) -> "PHEEncryptor":
        assert isinstance(self._inside, PHEEncryptor)
        return self._inside

    def unwrap_phe_tensor(self) -> "PHETensor":

        assert isinstance(self._inside, PHETensor)
        return self._inside

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> T:
        if check is not None and not check(self._inside):
            raise TypeError(f"`{self._inside}` check failed")
        return self._inside


class Futures(FuturesInterface):
    def __init__(self, insides) -> None:
        self._insides = insides

    def unwrap_tensors(self) -> List["FPTensor"]:

        for t in self._insides:
            assert isinstance(t, FPTensor)
        return self._insides

    def unwrap_phe_tensors(self) -> List["PHETensor"]:

        for t in self._insides:
            assert isinstance(t, PHETensor)
        return self._insides

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> List[T]:
        if check is not None:
            for i, t in enumerate(self._insides):
                if not check(t):
                    raise TypeError(f"{i}th element `{self._insides}` check failed")
        return self._insides


class GC:
    def __init__(self) -> None:
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

    def get_or_set_push_gc(self, key):
        if key not in self._push_gc_dict:
            self._push_gc_dict[key] = IterationGC()
        return self._push_gc_dict[key]

    def get_or_set_pull_gc(self, key):
        if key not in self._push_gc_dict:
            self._pull_gc_dict[key] = IterationGC()
        return self._pull_gc_dict[key]


class FederationParty(PartyInterface):
    def __init__(self, ctx, party: Tuple[str, str], namespace, gc: GC) -> None:
        self.ctx = ctx
        self.party = PartyMeta(party[0], party[1])
        self.namespace = namespace
        self.gc = gc

    def push(self, name: str, value):
        return _push(self.ctx, name, self.namespace, [self.party], self.gc, value)

    def pull(self, name: str) -> Future:
        return Future(_pull(self.ctx, name, self.namespace, [self.party], self.gc)[0])


class FederationParties(PartiesInterface):
    def __init__(self, ctx, parties: List[Tuple[str, str]], namespace: Namespace, gc: GC) -> None:
        self.ctx = ctx
        self.parties = [PartyMeta(party[0], party[1]) for party in parties]
        self.namespace = namespace
        self.gc = gc

    def __call__(self, key: int) -> FederationParty:
        return FederationParty(
            self.ctx, self.parties[key].as_tuple(), self.namespace, self.gc
        )

    def push(self, name: str, value):
        return _push(self.ctx, name, self.namespace, self.parties, self.gc, value)

    def pull(self, name: str) -> Futures:
        return Futures(_pull(self.ctx, name, self.namespace, self.parties, self.gc))


def _push(ctx, name: str, namespace: Namespace, parties: List[PartyMeta], gc: GC, value):
    if hasattr(value, "__federation_hook__"):
        value.__federation_hook__(ctx, name, parties)
    else:
        get_session().federation.remote(
            v=value,
            name=name,
            tag=namespace.fedeation_tag(),
            parties=parties,
            gc=gc.get_or_set_push_gc(name),
        )


def _pull(ctx, name: str, namespace: Namespace, parties: List[PartyMeta], gc: GC):
    raw_values = get_session().federation.get(
        name=name,
        tag=namespace.fedeation_tag(),
        parties=parties,
        gc=gc.get_or_set_pull_gc(name),
    )
    values = []
    for party, raw_value in zip(parties, raw_values):
        if isinstance(raw_value, FederationDeserializer):
            values.append(raw_value.do_deserialize(ctx, party))
        else:
            values.append(raw_value)
    return values


class _PartyUtil:
    def __init__(self) -> None:
        ...

    @classmethod
    def parse(
        cls,
        local_party: Tuple[Literal["guest", "host", "arbiter"], str],
        parties: Optional[List[Tuple[Literal["guest", "host", "arbiter"], str]]] = None,
    ) -> "_PartyUtil":
        ...

    def has_role(self, role: str):
        ...

    def role(self, role: str) -> List:
        ...

    def all_parties(self, ctx, namespace, gc) -> FederationParties:
        return FederationParties(ctx, self.parties, namespace, gc)

    def create_party(self, role: str, ctx, namespace, gc) -> FederationParty:
        if not self.has_role(role):
            raise RuntimeError(f"no {role} party has configurated")
        return FederationParty(ctx, self.role(role)[0], namespace, gc)

    def create_parties(self, role: str, ctx, namespace, gc) -> FederationParties:
        if not self.has_role(role):
            raise RuntimeError(f"no {role} party has configurated")
        return FederationParties(ctx, self.role(role), namespace, gc)

    @property
    def parties(self) -> List[Tuple[str, str]]:
        ...
