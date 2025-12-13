import logging
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# 1. Import your routers here
from app.api.routes import health, qsar
from app.config import settings
from app.core.database import init_database
from app.core.metrics import setup_metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Handles startup and shutdown logic (DB connections, Sentry, Model loading).
    """
    # --- STARTUP ---
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize Sentry if DSN is provided
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[
                FastApiIntegration(auto_enabling_instrumentations=False),
                LoggingIntegration(level=logging.INFO),
            ],
            traces_sample_rate=0.1,
            environment=settings.environment,
        )
        logger.info("Sentry initialized")

    # Initialize database connection
    try:
        await init_database()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # We don't raise here to allow the app to start even if DB fails (optional)
    
    logger.info("Application startup complete")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down application")


# Create FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="QSAR Analysis API for Drug Discovery & Toxicity Prediction",
    lifespan=lifespan,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their status codes."""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# Setup Prometheus metrics
setup_metrics(app)
logger.info("Prometheus metrics enabled")

# ============================================================================
# Register Routers
# ============================================================================

# Health check routes
app.include_router(health.router, prefix="/health", tags=["System Health"])

# 2. Register the QSAR Prediction routes
app.include_router(qsar.router, prefix="/qsar", tags=["QSAR Analysis"])


@app.get("/")
async def root():
    """Root endpoint providing API info."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "description": "QSAR Prediction Service for FDA Approval & Toxicity",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": "/docs"
    }
