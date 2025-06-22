from typing import Any

from blacksheep import Request, Response
from blacksheep.server import Application
from blacksheep.server.responses import text
from essentials.exceptions import (
    AcceptedException,
    ForbiddenException,
    NotImplementedException,
    ObjectNotFound,
    UnauthorizedException,
    InvalidArgument
)


def configure_error_handlers(app: Application) -> None:
    async def not_found_handler(
        app: Application, request: Request, exception: Exception
    ) -> Response:
        return text(str(exception) or "Not found", 404)

    async def not_implemented(*args: Any) -> Response:
        return text("Not implemented", status=500)

    async def unauthorized(*args: Any) -> Response:
        return text("Unauthorized", status=401)

    async def forbidden(*args: Any) -> Response:
        return text("Forbidden", status=403)

    async def accepted(*args: Any) -> Response:
        return text("Accepted", status=202)
    
    async def bad_request(*args: Any) -> Response:
        return Response(status=400, content={"success": False, "error": "The request body was not formed correctly."})

    app.exceptions_handlers.update(
        {
            ObjectNotFound: not_found_handler,
            NotImplementedException: not_implemented,
            UnauthorizedException: unauthorized,
            ForbiddenException: forbidden,
            AcceptedException: accepted,
            InvalidArgument: bad_request
        }
    )
