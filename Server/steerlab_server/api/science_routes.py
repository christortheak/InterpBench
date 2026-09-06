"""Shipped guidance, independent of workspace and compute/service topology."""
from fastapi import APIRouter, HTTPException
from ..experiment import science_catalog


def build_science_router():
    router = APIRouter()

    def read(owner, name):
        try:
            return owner(name)
        except science_catalog.ScienceRefusal as exc:
            raise HTTPException(status_code=400, detail={
                'code': exc.code, 'reason': str(exc), 'repairAction': exc.repair_action}) from exc

    @router.get('/api/science/catalog')
    def catalog():
        return science_catalog.catalog()

    @router.get('/api/science/guide/{method}')
    def guide(method: str):
        return read(science_catalog.guide, method)

    @router.get('/api/science/operation/{operation}')
    def operation(operation: str):
        return read(science_catalog.operation, operation)

    return router
