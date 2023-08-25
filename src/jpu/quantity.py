from typing import Generic

from pint.facets.plain import MagnitudeT, PlainQuantity


class JpuQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    pass
