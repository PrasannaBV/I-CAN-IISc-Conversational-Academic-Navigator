from aiohttp import web
import json

# Mock university course data
dummy_courses = {
    "CS101": {
        "title": "Intro to Computer Science",
        "credits": 3,
        "instructor": "Prof. Jane Doe",
        "semester": "Fall 2025"
    },
    "EE202": {
        "title": "Signals and Systems",
        "credits": 4,
        "instructor": "Dr. Alan Turing",
        "semester": "Spring 2026"
    }
}

async def handle(request):
    try:
        request_data = await request.json()
        
        # Basic JSON-RPC 2.0 validation
        if not all(k in request_data for k in ["jsonrpc", "method", "id"]):
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request"},
                "id": None
            }, status=400)
        
        # Handle the method call
        if request_data["method"] == "get_course_info":
            course_code = request_data.get("params", {}).get("course_code", "")
            code = course_code.strip().upper()
            
            if code in dummy_courses:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "result": dummy_courses[code],
                    "id": request_data["id"]
                })
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32602,
                        "message": "Course not found",
                        "data": f"Course code {code} not in database"
                    },
                    "id": request_data["id"]
                })
        else:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": request_data["id"]
            })
            
    except json.JSONDecodeError:
        return web.json_response({
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": "Parse error"},
            "id": None
        }, status=400)

app = web.Application()
app.router.add_post("/", handle)

if __name__ == "__main__":
    print(" MCP server running on http://localhost:5001")
    web.run_app(app, port=5001)
