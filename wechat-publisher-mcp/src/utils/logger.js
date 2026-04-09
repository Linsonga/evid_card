/**
 * 日志工具
 * 提供统一的日志记录功能，支持不同级别的日志输出
 */

//const logger = {
//  info: (message, ...args) => {
//    if (process.env.NODE_ENV !== 'production') {
//      console.log(`[INFO] ${message}`, ...args);
//    }
//  },
//
//  error: (message, error) => {
//    if (error && error.stack) {
//      console.error(`[ERROR] ${message}\n${error.stack}`);
//    } else {
//      console.error(`[ERROR] ${message}`, error || '');
//    }
//  },
//
//  debug: (message, ...args) => {
//    if (process.env.DEBUG) {
//      console.log(`[DEBUG] ${message}`, ...args);
//    }
//  },
//
//  warn: (message, ...args) => {
//    console.warn(`[WARN] ${message}`, ...args);
//  }
//};
//
//export default logger;


const logger = {
  info: (message, ...args) => {
    console.error(`[INFO] ${message}`, ...args); // 修改为 console.error
  },
  error: (message, error) => {
    console.error(`[ERROR] ${message}`, error || '');
  },
  debug: (message, ...args) => {
    if (process.env.DEBUG) {
      console.error(`[DEBUG] ${message}`, ...args); // 修改为 console.error
    }
  },
  warn: (message, ...args) => {
    console.error(`[WARN] ${message}`, ...args); // 修改为 console.error
  }
};
export default logger;