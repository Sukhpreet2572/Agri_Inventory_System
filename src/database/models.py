from datetime import datetime, date
import enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from src.database.connection import Base


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), unique=True, nullable=False, index=True)
    state = Column(String(100), nullable=False)
    district = Column(String(100), nullable=True)

    inventory_items = relationship("Inventory", back_populates="market")


class Commodity(Base):
    __tablename__ = "commodities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    unit = Column(String(30), default="Quintal (100 kg)")
    shelf_life_days = Column(Integer, default=30)

    inventory_items = relationship("Inventory", back_populates="commodity")


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, nullable=False)
    contact_email = Column(String(120), nullable=False)
    lead_time_days = Column(Integer, nullable=False, default=3)
    reliability_score = Column(Float, default=0.95)


class DailyPriceHistory(Base):
    """Stores both historical baseline prices and live daily prices synced from data.gov.in."""
    __tablename__ = "daily_price_history"
    __table_args__ = (UniqueConstraint("market_id", "commodity_id", "price_date", name="uq_market_comm_date"),)

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False, index=True)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False, index=True)
    price_date = Column(Date, nullable=False, index=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    modal_price = Column(Float, nullable=False)
    source = Column(String(30), default="HISTORICAL")  # 'HISTORICAL' or 'LIVE_GOV_API'


class Inventory(Base):
    """Tracks current warehouse stock, daily consumption, and reorder thresholds per Market & Commodity."""
    __tablename__ = "inventory"
    __table_args__ = (UniqueConstraint("market_id", "commodity_id", name="uq_inv_market_comm"),)

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)

    current_stock_qtl = Column(Float, nullable=False, default=100.0)
    avg_daily_demand_qtl = Column(Float, nullable=False, default=15.0)
    demand_std_qtl = Column(Float, nullable=False, default=4.0)
    safety_stock_qtl = Column(Float, nullable=False, default=20.0)
    reorder_point_qtl = Column(Float, nullable=False, default=65.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    market = relationship("Market", back_populates="inventory_items")
    commodity = relationship("Commodity", back_populates="inventory_items")
    supplier = relationship("Supplier")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)

    quantity_qtl = Column(Float, nullable=False)
    locked_unit_price = Column(Float, nullable=False)
    procurement_strategy = Column(String(50), default="STANDARD_REORDER")  # e.g. 'BULK_EARLY_BUY', 'MIN_JIT_BUY'
    status = Column(SAEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    order_date = Column(Date, default=date.today, nullable=False)
    expected_delivery_date = Column(Date, nullable=False)


class VendorSale(Base):
    """Logs outbound purchases made by downstream vendors/retailers from our warehouse."""
    __tablename__ = "vendor_sales"

    id = Column(Integer, primary_key=True, index=True)
    vendor_name = Column(String(120), nullable=False)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False)
    quantity_qtl = Column(Float, nullable=False)
    unit_sale_price = Column(Float, nullable=False)
    total_amount_inr = Column(Float, nullable=False)
    sale_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)